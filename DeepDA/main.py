import configargparse
from numpy.core.fromnumeric import argsort
import data_loader
import os
import torch
import models
import utils
from utils import str2bool, metric
import numpy as np
import random
import copy
import pretty_errors
from loss_funcs import MMDLoss, GNetGram

def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="config file path", default='/home/jindwang/mine/code/learning_to_match/DeepDA/DAN/DAN.yaml')
    parser.add("--seed", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    
    # network related
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)

    # data loading related
    parser.add_argument('--data_dir', type=str, default='/data/jindwang/OfficeHome')
    parser.add_argument('--src_domain', type=str, default='Art')
    parser.add_argument('--tgt_domain', type=str, default='Clipart')
    
    # training related
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=0, help="Early stopping")
    parser.add_argument('--epoch_based_training', type=str2bool, default=False, help="Epoch-based training / Iteration-based training")
    parser.add_argument("--n_iter_per_epoch", type=int, default=20, help="Used in Iteration-based training")

    # optimizer related
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # learning rate scheduler related
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=True)

    # transfer related
    parser.add_argument('--transfer_loss_weight', type=float, default=10)
    parser.add_argument('--transfer_loss', type=str, default='mmd')

    # metric related
    parser.add_argument('--metric', type=str, default='acc')

    # l2m related
    parser.add_argument('--gbatch', type=int, default=16)
    parser.add_argument('--meta_m',type=int, default=16)
    parser.add_argument('--glr', type=float, default=.0001)
    return parser

def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(args):
    '''
    src_domain, tgt_domain data to load
    '''
    folder_src = os.path.join(args.data_dir, args.src_domain)
    folder_tgt = os.path.join(args.data_dir, args.tgt_domain)
    source_loader, n_class = data_loader.load_data(
        folder_src, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True, num_workers=args.num_workers)
    target_train_loader, _ = data_loader.load_data(
        folder_tgt, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True, num_workers=args.num_workers)
    target_test_loader, _ = data_loader.load_data(
        folder_tgt, args.batch_size, infinite_data_loader=False, train=False, num_workers=args.num_workers)
    return source_loader, target_train_loader, target_test_loader, n_class

def get_model(args):
    model = models.TransferNet(
        args.n_class, transfer_loss=args.transfer_loss, base_net=args.backbone, max_iter=args.max_iter, use_bottleneck=args.use_bottleneck).to(args.device)
    return model

def get_optimizer(model, args):
    initial_lr = args.lr if not args.lr_scheduler else 1.0
    params = model.get_parameters(initial_lr=initial_lr)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    return optimizer

def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    return scheduler

def inference(model, loader):
    model.eval()
    all_probs, all_labels, all_preds = None, None, None
    first_test = True
    with torch.no_grad():
        for data, label in loader:
            data, label = data.cuda(), label.cuda()
            _, probs, preds = model.predict(data)
            if first_test:
                all_probs = probs.data.float()
                all_labels = label.data.float()
                all_preds = preds.data.float()
                first_test = False
            else:
                all_probs = torch.cat((all_probs, probs), 0)
                all_labels = torch.cat((all_labels, label), 0)
                all_preds = torch.cat((all_preds, preds), 0)
    return all_probs, all_preds, all_labels

def test(model, loader, calc_f1=False):
    all_probs, all_preds, all_labels = inference(model, loader)
    accuracy = torch.sum(
        torch.squeeze(all_preds).float() == all_labels).item() / float(
            all_labels.size()[0])
    ret = {}
    ret["acc"] = accuracy
    if calc_f1:
        ret = metric(all_labels.cpu().detach().numpy(), all_preds.cpu(
        ).detach().numpy(), all_probs.cpu().detach().numpy())
    return ret

def train(source_loader, target_train_loader, target_test_loader, model, optimizer, lr_scheduler, args):
    kwargs = {}
    kwargs['n_input'] = 16 * 16
    kwargs['n_hiddens'] = [128, 64]
    kwargs['n_output'] = 1
    kwargs['use_set'] = True
    kwargs['drop_out'] = 0
    kwargs['mono'] = True
    kwargs['init_net'] = True
    gnet = GNetGram(**kwargs)

    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    n_batch = min(len_source_loader, len_target_loader)
    if n_batch == 0:
        n_batch = args.n_iter_per_epoch
    iter_source, iter_target = iter(source_loader), iter(target_train_loader)

    best_acc = 0
    best_all = {}
    stop = 0
    log = []
    cls_criterion = torch.nn.CrossEntropyLoss()

    optimizer_g = torch.optim.SGD(gnet.net.parameters(), lr=args.glr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    model_old = args.model_old
    for e in range(1, args.n_epoch+1):
        model_old.load_state_dict(model.state_dict())
        model_old.train()
        model.train()
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        model.epoch_based_processing(n_batch)
        model_old.epoch_based_processing(n_batch)

        if max(len_target_loader, len_source_loader) != 0:
            iter_source, iter_target = iter(source_loader), iter(target_train_loader)
        
        # update main model
        for _ in range(n_batch):
            data_source, label_source = next(iter_source) # .next()
            data_target, _ = next(iter_target) # .next()
            data_source, label_source = data_source.to(
                args.device), label_source.to(args.device)
            data_target = data_target.to(args.device)

            clf_loss, transfer_loss = model(data_source, data_target, label_source)
            loss = clf_loss + args.transfer_loss_weight * transfer_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())

        gloss_total = utils.AverageMeter()
        folder_src = os.path.join(args.data_dir, args.src_domain)
        folder_tgt = os.path.join(args.data_dir, args.tgt_domain)
        source_loader1, n_class = data_loader.load_data(
            folder_src, args.batch_size, infinite_data_loader=False, train=True, num_workers=args.num_workers)
        target_train_loader1, _ = data_loader.load_data(
            folder_tgt, args.batch_size, infinite_data_loader=False, train=False, num_workers=args.num_workers)
        # update gnet

        if e >= 0:
            meta_src = generate_metadata_soft(
                args.meta_m, source_loader1, model, args.gbatch, select_mode='top', source=True)
            meta_tar = generate_metadata_soft(
                args.meta_m, target_train_loader1, model, args.gbatch, select_mode='top')
            for _ in range(1):
                for datas, datat in zip(meta_src, meta_tar):
                    (xs, ys), (xt, yt) = datas, datat
                    xs, ys, xt, yt = xs.cuda(), ys.cuda(), xt.cuda(), yt.cuda()
                    feat_old = model_old.get_features(torch.cat((xs, xt), dim=0))
                    feat_new = model.get_features(torch.cat((xs, xt), dim=0))
                    loss_old = gnet(feat_old[:feat_old.size(0) // 2], feat_old[feat_old.size(0) // 2:])
                    loss_new = gnet(feat_old[:feat_new.size(0) // 2], feat_old[feat_new.size(0) // 2:])
                    gloss = gnet_diff(loss_old, loss_new)
                    optimizer_g.zero_grad()
                    gloss.backward()
                    optimizer_g.step()
                    gloss_total.update(gloss.item())

            
        log.append([train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg])
        
        info = 'Epoch: [{:2d}/{}], cls_loss: {:.4f}, transfer_loss: {:.4f}, total_Loss: {:.4f}, gloss: {:.6f}'.format(
                        e, args.n_epoch, train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg, gloss_total.avg)
        # Test
        stop += 1
        test_res = test(model, target_test_loader, calc_f1=True if args.metric == 'f1' else False)
        info += ', test: ' + str(test_res)
        test_acc = test_res['acc']
        if best_acc < test_acc:
            best_acc = test_acc
            best_all = test_res
            stop = 0
        if args.early_stop > 0 and stop >= args.early_stop:
            print(info)
            break
        print(info)
    print('Transfer result: ' + str(best_all))

def gnet_diff(g_loss_pre, g_loss_post):
    """Difference between previous and current gnet losses: tanh(g_loss_pre, g_loss_post)
    May use different functions such as subtraction, log, or crossentropy...

    Args:
        g_loss_pre (float): previous gnet loss
        g_loss_post (float): current gnet loss

    Returns:
        diff (float): difference
    """
    diff = g_loss_post - g_loss_pre
    # diff = g_loss_post - g_loss_pre
    # diff = torch.tanh(g_loss_post - g_loss_pre)
    # diff = torch.tanh(torch.nn.CrossEntropyLoss()(g_loss_pre, g_loss_post))
    # diff = torch.log(g_loss_pre) - torch.log(g_loss_post)
    return diff


    

def generate_metadata_soft(m, loader, model, batch_size, select_mode='top', source=False):
        """Generate meta data with soft labels

        Args:
            m (int): # meta data for each class
            loader (dataloader): data loader
            model (model): model
            select_mode (str): "random" for random sampling or "top" for sampling top m
            source (bool): denotes whether the loader is for source (True) or not (False)

        Returns:
            loader: metaloader
        """
        test_imgs = loader.dataset.imgs
        cls_idx = loader.dataset.class_to_idx
        if not source:
            prob, softlabels, _ = inference(model, loader)
            prob, softlabels = prob.cpu().detach().numpy(), softlabels.cpu().detach().numpy()
        else:  # source domain, then soft labels are groundtruth, prob is all 1
            softlabels = np.array([int(item[1]) for item in test_imgs])
            prob = np.array([1] * len(softlabels))
        test_imgs_path = [item[0] for item in test_imgs]
        test_imgs_index = np.arange(len(test_imgs_path))
        test_imgs_all = np.hstack((
            test_imgs_index.reshape((len(test_imgs_index), 1)),
            softlabels.reshape((len(softlabels), 1)),
            prob.reshape((len(softlabels), 1))))
        threshold = 0.8
        confident_imgs = test_imgs_all[prob >= threshold]
        rest_imgs = test_imgs_all[prob < threshold]
        imgs_select_all = []

        # find the m samples for each class to contruct meta loader
        for cls in loader.dataset.classes:
            cls = int(cls_idx[cls])
            imgs_cls = confident_imgs[confident_imgs[:, 1] == cls]
            if len(imgs_cls) >= m:
                if select_mode == 'random':
                    np.random.shuffle(imgs_cls)
                    imgs_select_cls = imgs_cls[:m]
                elif select_mode == 'top':
                    probs_conf = imgs_cls[:, 2]
                    ind = probs_conf.argsort()
                    imgs_select_cls = imgs_cls[ind[::-1]]
                    imgs_select_cls = imgs_select_cls[:m]
            elif len(imgs_cls) > 0:
                imgs_select_cls = imgs_cls
                np.random.shuffle(rest_imgs)
                rest = rest_imgs[:m - len(imgs_cls)]
                imgs_select_cls = np.vstack((imgs_select_cls, rest))
                imgs_select_cls[:, 1] = cls
            else:
                probs_conf = rest_imgs[:, 2]
                ind = probs_conf.argsort()
                imgs_select_cls = rest_imgs[ind]
                imgs_select_cls = imgs_select_cls[:m]
                imgs_select_cls[:, 1] = cls
            imgs_select = [(test_imgs_path[int(item[0])], int(item[1]))
                           for item in imgs_select_cls]
            imgs_select_all.extend(imgs_select)
        meta_dataset = data_loader.MetaDataset(imgs_select_all)
        meta_loader = data_loader.load_metadata(meta_dataset, batch_size=batch_size)
        return meta_loader

def main():
    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(args)
    set_random_seed(args.seed)
    source_loader, target_train_loader, target_test_loader, n_class = load_data(args)
    setattr(args, "n_class", n_class)
    if args.epoch_based_training:
        setattr(args, "max_iter", args.n_epoch * min(len(source_loader), len(target_train_loader)))
    else:
        setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
    model = get_model(args)
    args.model_old = get_model(args)
    optimizer = get_optimizer(model, args)
    
    if args.lr_scheduler:
        scheduler = get_scheduler(optimizer, args)
    else:
        scheduler = None
    train(source_loader, target_train_loader, target_test_loader, model, optimizer, scheduler, args)

if __name__ == "__main__":
    main()
