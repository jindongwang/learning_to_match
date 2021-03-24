import torch
import numpy as np
import copy
import datetime
import os
import math

from utils.helper import metric, AverageMeter
import copy
from utils.visualize import Visualize
import data_loader
from utils.helper import set_random

binary_data = ['covid-19', 'covid', 'covid19', 'visda-binary', 'vbinary', 'bac', 'viral']

class L2MTrainer(object):
    """Main training class"""

    def __init__(self, gnet, model, model_old, dataloaders, config) -> None:
        """Init func

        Args:
            gnet (GNet): gnet
            model (model): model
            model_old (old model): old model
            dataloaders (list): [source_train_loader, target_train_loader, target_test_loader]
            config (dict): config dict
        """
        self.gnet = gnet
        self.model = model
        self.model_old = model_old
        self.config = config
        if self.config.gopt == 'adam':
            self.optimizer_g = torch.optim.Adam(
                self.gnet.parameters(), lr=self.config.glr)
        elif self.config.gopt == 'sgd':
            self.optimizer_g = torch.optim.SGD(self.gnet.parameters(
            ), lr=self.config.glr, momentum=self.config.momentum, weight_decay=self.config.weight_decay)

        self.max_iter = self.config.nepoch
        self.train_source_loader, self.train_target_loader, self.test_target_loader = dataloaders
        self.meta_loader = None

        self.save_path = self.config.save_path
        self.meta_m = self.config.meta_m

    def train(self):
        self.pprint("Start train...")
        stop = 0
        mxacc = 0
        best_res = {}
        for epoch in range(self.max_iter):
            stop += 1
            self.model_old.load_state_dict(self.model.state_dict())
            self.gnet.train()
            self.model_old.train()
            self.model.train()

            self.get_optimizer(epoch)

            # update main model
            cls_loss = 0
            lambd = 2 / (1 + math.exp(-10 * (epoch) / self.max_iter)) - 1
            cls_loss, gloss = self.update_main_model(self.train_source_loader, self.train_target_loader, lambd)

            # update gnet
            # construct meta_loader from target_loader before each epoch
            diff, g_loss_pre, g_loss_post, mar_loss, mar_loss2 = 0, 0, 0, 0, 0
            if epoch > 5 and self.config.mu > 0:
                self.load_meta_data()
                diff, g_loss_pre, g_loss_post, mar_loss, mar_loss2 = self.update_gnet(
                    self.meta_src, self.meta_tar)

            
            calcf1 = True if self.config.dataset.lower() in binary_data else False
            ret = self.evaluate(self.model, self.test_target_loader, calcf1=calcf1)
            acc = ret["f1"] if calcf1 else ret["accuracy"]

            if acc >= mxacc:
                stop = 0
                mxacc = acc
                torch.save(self.model.state_dict(), self.save_path.replace('.log', '.pkl'))
                best_res = ret
            if not calcf1:
                self.pprint(f"[Epoch:{epoch:02d}]: cls_loss: {cls_loss:.5f}, g_loss: {diff:.10f}, acc:{acc:.4f}, mxacc:{mxacc:.4f}")
            else:
                self.pprint(f"[Epoch:{epoch:02d}]: cls_loss: {cls_loss:.5f}, g_loss: {diff:.10f}, mxf1: {mxacc:.4f}")
                self.pprint(f"P: {ret['p']:.4f}, R: {ret['r']:.4f}, f1: {ret['f1']:.4f}, acc: {ret['accuracy']:.4f}, auc: {ret['auc']:.4f}")

            if stop >= self.config.early_stop:
                self.pprint("=================Early stop!!")
                break
        if calcf1:
            self.pprint(
                f"P: {best_res['p']:.4f}, R: {best_res['r']:.4f}, f1: {best_res['f1']:.4f}, acc: {best_res['accuracy']:.4f}, auc: {best_res['auc']:.4f}")
        else:
            self.pprint(f"acc: {best_res['accuracy']:.4f}")

    def update_main_model(self, src_train_loader, tar_train_loader, lambd):
        """Update the main model (feature learning)

        Args:
            src_train_loader (dataloader): Source dataloader
            tar_train_loader (dataloader): Target dataloader

        Returns:
            tuple: classification loss and gloss (on average)
        """
        lst_cls, lst_gloss = AverageMeter('loss_cls'), AverageMeter('gloss')
        for (datas, datat) in zip(src_train_loader, tar_train_loader):
            (xs, ys), (xt, _) = datas, datat
            if xs.size(0) != xt.size(0):
                continue

            xs, xt, ys = xs.cuda(), xt.cuda(), ys.cuda()
            feat, _, _, classifier_loss, mar_loss, _ = self.model(torch.cat((xs, xt), dim=0), ys)
            total_loss = classifier_loss
            if self.config.mu > 0:
                gloss = self.gnet(feat)
                total_loss = total_loss + self.config.mu * gloss
                lst_gloss.update(gloss.item())
            if self.config.lamb > 0:
                total_loss = total_loss + self.config.lamb * mar_loss

            self.optimizer_m.zero_grad()
            total_loss.backward()
            self.optimizer_m.step()

            lst_cls.update(classifier_loss.item())
            
        return lst_cls.avg, lst_gloss.avg

    def update_gnet(self, source_loader, meta_loader, nepoch=1):
        """Update gnet

        Args:
            source_loader (dataloader): source dataloader
            meta_loader (dataloader): metaloader
            nepoch (int, optional): update g for how much epochs. Defaults to 1.

        Returns:
            tuple: several losses
        """
        lst_diff, lst_g_loss_pre, lst_g_loss_post, lst_mar_loss, lst_mar_loss2 = AverageMeter('diff'), AverageMeter('g_loss_pre'), AverageMeter('g_loss_post'), AverageMeter('mar_loss'), AverageMeter('mar_loss_2')
        for _ in range(nepoch):
            for (datas, datat) in zip(source_loader, meta_loader):
                (xs, ys), (x_meta, y_meta) = datas, datat
                xs, ys, x_meta, y_meta = xs.cuda(), ys.cuda(), x_meta.cuda(), y_meta.cuda()
                if xs.size(0) != x_meta.size(0):
                    continue
                inputs_tmp = torch.cat((xs, x_meta), dim=0)

                feat, _, _, _, mar_loss, _ = self.model_old(inputs_tmp, ys)
                g_loss_pre = self.gnet(feat)

                feat2, _, _, _, mar_loss2, _ = self.model(inputs_tmp, ys)
                g_loss_post = self.gnet(feat2)

                diff = gnet_diff(g_loss_pre, g_loss_post)

                self.optimizer_g.zero_grad()
                diff.backward()
                self.optimizer_g.step()

                lst_diff.update(diff.item())
                lst_g_loss_pre.update(g_loss_pre.item())
                lst_g_loss_post.update(g_loss_post.item())
                lst_mar_loss.update(mar_loss.item())
                lst_mar_loss2.update(mar_loss2.item())
        return lst_diff.avg, lst_g_loss_pre.avg, lst_g_loss_post.avg, lst_mar_loss.avg, lst_mar_loss2.avg

    def get_optimizer(self, epoch):
        lr = self.config.init_lr / math.pow((1 + 10 * epoch / self.max_iter), self.config.gamma)
        if self.model.use_adv:
            self.optimizer_m = torch.optim.SGD([
                {'params': self.model.featurizer.parameters()},
                {'params': self.model.bottleneck_layer.parameters(), 'lr': lr},
                {'params': self.model.classifier_layer.parameters(), 'lr': lr},
                {'params': self.model.domain_classifier.parameters(), 'lr': lr},
                {'params': self.model.domain_classifier_class.parameters(), 'lr': lr},
            ], lr=lr/10, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        else:
            self.optimizer_m = torch.optim.SGD([
                {'params': self.model.featurizer.parameters()},
                {'params': self.model.bottleneck_layer.parameters(), 'lr': lr},
                {'params': self.model.classifier_layer.parameters(), 'lr': lr}
            ], lr=lr/10, momentum=self.config.momentum, weight_decay=self.config.weight_decay)

    def load_meta_data(self):
        kwargs = {'num_workers': 4, 'pin_memory': True}
        train_val_split = -1
        # set_random(self.config.seed * 2)
        # self.meta_source_loader = data_loader.load_testing(
        #     self.config.data_path, self.config.src, self.config.batch_size, kwargs)
        # self.train_source_loader = data_loader.load_training(
        #     self.config.data_path, self.config.src, self.config.batch_size, kwargs, train_val_split)
        # self.train_target_loader = data_loader.load_training(
        #     self.config.data_path, self.config.tar, self.config.batch_size, kwargs, train_val_split)

        
        self.meta_src = self.generate_metadata_soft(
            self.meta_m, self.train_source_loader, self.model, self.config.gbatch, select_mode='top', source=True)
        self.meta_tar = self.generate_metadata_soft(
            self.meta_m, self.test_target_loader, self.model, self.config.gbatch, select_mode='top')

    def generate_metadata_soft(self, m, loader, model, batch_size, select_mode='top', source=False):
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
            prob, softlabels, _ = self.inference(model, loader)
            prob, softlabels = prob.cpu().detach().numpy(), softlabels.cpu().detach().numpy()
        else:  # source domain, then soft labels are groundtruth, prob is all 1
            softlabels = np.array([int(cls_idx[str(item[1])]) for item in test_imgs])
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

    def inference(self, model, loader):
        model.eval()
        all_probs, all_labels, all_preds = None, None, None
        first_test = True
        with torch.no_grad():
            for data, label in loader:
                data, label = data.cuda(), label.cuda()
                probs, _, preds = model.predict(data)
                if first_test:
                    all_probs = probs.data.float()
                    all_labels = label.data.float()
                    all_preds = preds.data.float()
                    first_test = False
                else:
                    all_probs = torch.cat((all_probs, probs), 0)
                    all_labels = torch.cat((all_labels, label), 0)
                    all_preds = torch.cat((all_preds, preds), 0)
        model.train()
        return all_probs, all_preds, all_labels

    def evaluate(self, model, input_loader, calcf1=False):
        all_probs, all_preds, all_labels = self.inference(model, input_loader)
        accuracy = torch.sum(
            torch.squeeze(all_preds).float() == all_labels).item() / float(
                all_labels.size()[0])
        ret = {}
        ret["accuracy"] = accuracy
        if calcf1:
            ret = metric(all_labels.cpu().detach().numpy(), all_preds.cpu().detach().numpy(), all_probs.cpu().detach().numpy())
        return ret

    def pprint(self, *text):
        # print with UTC+8 time
        log_file = self.config.save_path.replace(".pkl", ".log")
        curr_time = ("[" + str(datetime.datetime.utcnow() +
                               datetime.timedelta(hours=8))[:19] + "] -")
        print(curr_time, *text, flush=True)
        if not os.path.exists(log_file):
            if not os.path.exists(self.config.save_folder):
                os.mkdir(self.config.save_folder)
            fp = open(self.config.log_file, "w")
            fp.close()
        with open(log_file, "a") as f:
            print(curr_time, *text, flush=True, file=f)


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


def update_params(model, grads, lr):
    """Update params by gradient descent

    Args:
        model (nn.Module): a pytorch model
        grads (list): a list of grads
        lr (float): learning rate
    """
    for idx, f in enumerate(model.parameters()):
        if grads[idx] is not None:
            f.data.sub_(lr * grads[idx])


def get_grad(model):
    grad_phi = []
    for _, (k, v) in enumerate(model.state_dict().items()):
        if k.__contains__('bn'):
            grad_phi.append(None)
        else:
            grad_phi.append(v.grad)
    return grad_phi

 
def generate_metadata(m, loader):
    targetimgs = loader.dataset.imgs
    start_idx = []
    idx_vis = []
    for idd, i in enumerate(targetimgs):
        idx = i[1]
        if idx not in idx_vis:
            start_idx.append(idd)
            idx_vis.append(idx)
    metaimgs = []
    for c in range(len(start_idx)):
        if c != len(start_idx) - 1:
            shuffleimgs = targetimgs[start_idx[c]:start_idx[c + 1]]
        else:
            shuffleimgs = targetimgs[start_idx[c]:]
        np.random.shuffle(shuffleimgs)
        metaimgs.extend(shuffleimgs[:m])
    meta_loader = copy.deepcopy(loader)
    meta_loader.dataset.imgs = metaimgs
    return meta_loader


def set_require_grad(model, grad=True):
    for item in model.parameters():
        item.requires_grad = grad
