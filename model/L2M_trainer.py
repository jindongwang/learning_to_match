import itertools
from numpy.core.fromnumeric import argsort
import torch
import numpy as np
import copy
import datetime
import os

from torch.utils.data import dataloader
from data_loader import MetaDataset
from utils.utils_f1 import metric
import copy
from utils.visualize import Visualize
import data_loader
import random


class L2MTrainer(object):
    """Main training class"""

    def __init__(self, gnet, model, model_old, dataloaders, config, max_iter=200000) -> None:
        """Init func

        Args:
            gnet (GNet): gnet
            model (model): model
            model_old (old model): old model
            dataloaders (list): [source_train_loader, target_train_loader, target_test_loader]
            config (dict): config dict
            max_iter (int, optional): maximum iteration num. Defaults to 200000.
        """
        self.gnet = gnet
        self.model = model
        self.model_old = model_old
        self.config = config

        self.param_groups = self.model.get_parameter_list()
        self.group_ratios = [group["lr"] for group in self.param_groups]
        self.optimizer_m = torch.optim.SGD(self.param_groups, lr=self.config.init_lr, momentum=self.config.momentum,
                                           weight_decay=self.config.weight_decay, nesterov=self.config.nesterov)
        if self.config.gopt == 'adam':
            self.optimizer_g = torch.optim.Adam(
                self.gnet.parameters(), lr=self.config.glr)
        elif self.config.gopt == 'sgd':
            self.optimizer_g = torch.optim.SGD(self.gnet.parameters(
            ), lr=self.config.glr, momentum=self.config.momentum, weight_decay=self.config.weight_decay)

        self.max_iter = max_iter
        self.train_source_loader, self.train_target_loader, self.test_target_loader = dataloaders
        self.meta_loader = None

        self.save_path = self.config.save_path
        self.meta_m = self.config.meta_m

        self.lr_scheduler = INVScheduler(
            gamma=self.config.gamma, decay_rate=self.config.decay_rate, init_lr=self.config.init_lr)
        self.vis = Visualize(port=8097, env=self.config.exp)
        self.iter_num = 0
        self.a = torch.randn(128, 1024).cuda()

    def generate_metadata_soft(self, m, loader, model, select_mode='random', source=False):
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
        prob, softlabels = get_softlabel(model, loader)
        prob, softlabels = prob.cpu().detach().numpy(), softlabels.cpu().detach().numpy()
        test_imgs = loader.dataset.imgs
        cls_idx = loader.dataset.class_to_idx
        test_imgs_path = [item[0] for item in test_imgs]
        if source: # if source, then soft labels=true labels
            softlabels = np.array([int(cls_idx[item[1]]) for item in test_imgs])
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
                    probs_conf = imgs_cls[:,2]
                    ind = probs_conf.argsort()
                    imgs_select_cls = imgs_cls[ind[::-1]]
                    imgs_select_cls = imgs_select_cls[:m]
            elif len(imgs_cls) > 0:
                imgs_select_cls = imgs_cls
                np.random.shuffle(rest_imgs)
                imgs_select_cls = np.vstack(
                    (imgs_select_cls, rest_imgs[:m - len(imgs_cls)]))
                imgs_select_cls[:, 1] = cls
            else:
                np.random.shuffle(rest_imgs)
                imgs_select_cls = rest_imgs[:m]
                imgs_select_cls[:, 1] = cls
            imgs_select = [(test_imgs_path[int(item[0])], int(item[1]))
                           for item in imgs_select_cls]
            imgs_select_all.extend(imgs_select)
        meta_dataset = data_loader.MetaDataset(imgs_select_all)
        meta_loader = data_loader.load_metadata(
            meta_dataset, batch_size=self.config.batch_size)
        return meta_loader

    def update_main_model(self, src_train_loader, tar_train_loader):
        """Update the main model (feature learning)

        Args:
            src_train_loader (dataloader): Source dataloader
            tar_train_loader (dataloader): Target dataloader

        Returns:
            tuple: classification loss and gloss (on average)
        """
        classifier_loss, gloss = -1, -1
        lst_cls, lst_gloss = [], []
        for (datas, datat) in zip(src_train_loader, tar_train_loader):
            inputs_source, labels_source = datas
            inputs_target, _ = datat
            if inputs_source.size(0) != inputs_target.size(0):
                continue

            self.optimizer_m, _ = self.lr_scheduler.next_optimizer(
                self.group_ratios, self.optimizer_m, self.iter_num / 5)
            inputs_source, inputs_target, labels_source = inputs_source.cuda(
            ), inputs_target.cuda(), labels_source.cuda(),
            inputs = torch.cat((inputs_source, inputs_target), dim=0)
            feat, logits, _, _, _, classifier_loss, mar_loss, cond_loss = self.model(
                inputs, labels_source)
            m_feat = self.model.match_feat(cond_loss, mar_loss, feat, logits)

            gloss = self.gnet(m_feat).mean()
            total_loss = classifier_loss + gloss

            self.optimizer_m.zero_grad()
            total_loss.backward()
            self.optimizer_m.step()
            lst_cls.append(classifier_loss.item())
            lst_gloss.append(gloss.item())
            self.iter_num += 1
        avg_loss = np.array(lst_cls).mean()
        avg_gloss = np.array(lst_gloss).mean()
        return avg_loss, avg_gloss

    def update_gnet(self, source_loader, meta_loader, nepoch=1):
        """Update gnet

        Args:
            source_loader (dataloader): source dataloader
            meta_loader (dataloader): metaloader
            nepoch (int, optional): update g for how much epochs. Defaults to 1.

        Returns:
            tuple: several losses
        """
        lst_diff, lst_g_loss_pre, lst_g_loss_post, lst_mar_loss, lst_mar_loss2 = [], [], [], [], []
        for _ in range(nepoch):
            for (datas, datat) in zip(source_loader, meta_loader):
                inputs_source, labels_source = datas
                meta_data, meta_label = datat
                meta_data, meta_label = meta_data.cuda(), meta_label.cuda()
                inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
                if inputs_source.size(0) != meta_data.size(0):
                    continue
                inputs_tmp = torch.cat((inputs_source, meta_data), dim=0)

                feat, logits, _, _, _, classifier_loss, mar_loss, cond_loss = self.model_old(
                    inputs_tmp, labels_source)
                m_feat = self.model_old.match_feat(
                    cond_loss, mar_loss, feat, logits)
                g_loss_pre = self.gnet(m_feat).mean()

                feat2, logits2, _, _, _, classifier_loss2, mar_loss2, cond_loss2 = self.model(
                    inputs_tmp, labels_source)
                m_feat2 = self.model.match_feat(
                    cond_loss2, mar_loss2, feat2, logits2)
                g_loss_post = self.gnet(m_feat2).mean()

                diff = gnet_diff(g_loss_pre, g_loss_post)
                self.optimizer_g.zero_grad()
                diff.backward()
                self.optimizer_g.step()
                # update_params(self.gnet, get_grad(self.gnet), self.config.glr)

                lst_diff.append(diff.item())
                lst_g_loss_pre.append(g_loss_pre.item())
                lst_g_loss_post.append(g_loss_post.item())
                lst_mar_loss.append(mar_loss.item())
                lst_mar_loss2.append(mar_loss2.item())
        return np.array(lst_diff).mean(), np.array(lst_g_loss_pre).mean(), np.array(lst_g_loss_post).mean(), np.array(lst_mar_loss).mean(), np.array(lst_mar_loss2).mean()

    def train(self):
        self.pprint("Start train...")
        iter_num = 0
        epoch = 0
        stop = 0
        mxacc = 0
        best_res = {}
        while True:
            self.model_old.load_state_dict(self.model.state_dict())
            self.gnet.train()
            self.model_old.train()
            self.model.train()

            # construct meta_loader from target_loader before each epoch
            if epoch == 0:
                self.meta_loader = generate_metadata(
                    m=self.meta_m, loader=self.test_target_loader)
            else:
                kwargs = {'num_workers': 4, 'pin_memory': True}
                train_val_split = -1
                self.config.seed = epoch * 2
                np.random.seed(self.config.seed)
                torch.manual_seed(self.config.seed)
                random.seed(self.config.seed)
                torch.cuda.manual_seed(self.config.seed)
                torch.cuda.manual_seed_all(self.config.seed)
                self.meta_source_loader = data_loader.load_testing(
                    self.config.root_path, self.config.source_dir, self.config.batch_size, kwargs)
                self.train_source_loader = data_loader.load_training(
                    self.config.root_path, self.config.source_dir, self.config.batch_size, kwargs, train_val_split)
                self.train_target_loader = data_loader.load_training(
                    self.config.root_path, self.config.test_dir, self.config.batch_size, kwargs, train_val_split)

                self.meta_loader = self.generate_metadata_soft(
                    self.meta_m, self.test_target_loader, model=self.model, select_mode='top')
                self.meta_source = self.generate_metadata_soft(
                    self.meta_m, self.meta_source_loader, model=self.model, select_mode='top')

            # update main modeld
            cls_loss = 0
            cls_loss, gloss = self.update_main_model(
                self.train_source_loader, self.train_target_loader)

            # update gnet
            diff, g_loss_pre, g_loss_post, mar_loss, mar_loss2 = 0, 0, 0, 0, 0
            if epoch > 0:
                diff, g_loss_pre, g_loss_post, mar_loss, mar_loss2 = self.update_gnet(
                    self.meta_source, self.meta_loader)

            self.vis.plot_line([mar_loss, mar_loss2], epoch,
                               title="MMD", legend=["old", "new"])
            self.vis.plot_line([g_loss_pre, g_loss_post],
                               epoch, title="GNet", legend=["old", "new"])

            stop += 1
            calcf1 = True if self.config.dataset.lower(
            ) in ['covid-19', 'covid', 'covid19'] else False
            ret = evaluate(self.model, self.test_target_loader, calcf1=calcf1)
            acc = ret["f1"] if calcf1 else ret["accuracy"]
            
            if acc >= mxacc:
                stop = 0
                mxacc = acc
                torch.save(self.model.state_dict(),
                           self.save_path.replace('.log', '.mdl'))
                best_res = ret
            if not calcf1:
                self.pprint(
                    f"[Epoch:{epoch:02d}], cls_loss: {cls_loss:.5f}, g_loss: {diff:.10f}, acc:{acc:.4f}, mxacc:{mxacc:.4f}")
            else:
                self.pprint(
                    f"[Epoch:{epoch:02d}], cls_loss: {cls_loss:.5f}, g_loss: {diff:.10f}, mxf1: {mxacc:.4f}")
                self.pprint(
                    f"P: {ret['p']:.4f}, R: {ret['r']:.4f}, f1: {ret['f1']:.4f}, acc: {ret['accuracy']:.4f}, auc: {ret['auc']:.4f}")
            epoch += 1

            if iter_num >= self.max_iter:
                self.pprint("finish train")
                break
            if stop >= self.config.early_stop:
                self.pprint("=================Early stop!!")
                break
        self.pprint(f"Max result: {mxacc}")
        self.pprint(best_res)
        self.pprint("Train is finished!")

    def pprint(self, *text):
        # print with UTC+8 time
        log_file = self.config.save_path.replace(".mdl", ".log")
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


class INVScheduler(object):
    def __init__(self, gamma, decay_rate, init_lr=0.001):
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.init_lr = init_lr

    def next_optimizer(self, group_ratios, optimizer, num_iter):
        lr = self.init_lr * (1 + self.gamma * num_iter)**(-self.decay_rate)
        i = 0
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * group_ratios[i]
            i += 1
        return optimizer, lr


def get_softlabel(model, loader):
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
    return all_probs, all_preds


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


def evaluate(model, input_loader, calcf1=False):
    model.eval()
    all_probs, all_labels, all_preds = None, None, None
    first_test = True
    with torch.no_grad():
        for data, label in input_loader:
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
    accuracy = torch.sum(
        torch.squeeze(all_preds).float() == all_labels).item() / float(
            all_labels.size()[0])
    ret = {}
    ret["accuracy"] = accuracy
    if calcf1:
        ret = metric(all_labels.cpu().detach().numpy(), all_preds.cpu(
        ).detach().numpy(), all_probs.cpu().detach().numpy())
    return ret


if __name__ == "__main__":
    a = L2MTrainer(None, None, None, None, None, None, None, None)
    print(a)
