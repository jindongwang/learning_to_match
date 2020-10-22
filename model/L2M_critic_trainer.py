import torch
import numpy as np
import copy
import datetime
import os
from utils.utils_f1 import metric
from utils.visualize import Visualize
import copy
import data_loader
from model import L2M


def update_params(model, grads, lr):
    for idx, f in enumerate(model.parameters()):
        if grads[idx] is not None:
            f.sub_(lr * grads[idx])


def gnet_diff(g_loss_pre, g_loss_post):
    # diff = g_loss_post - g_loss_pre
    diff = torch.tanh(g_loss_post - g_loss_pre)
    # diff = torch.tanh(torch.nn.CrossEntropyLoss()(g_loss_pre, g_loss_post))
    # diff = torch.log(g_loss_pre) - torch.log(g_loss_post)
    return diff


class L2MTrainerCritic(object):
    def __init__(self, gnet, model, model_old, optimizer_m, optimizer_g, dataloaders, config, max_iter=200000) -> None:
        self.gnet = gnet
        self.model = model
        self.model_old = model_old
        self.optimizer_m = optimizer_m
        self.optimizer_g = optimizer_g
        self.max_iter = max_iter
        self.train_source_loader, self.train_target_loader, self.test_target_loader = dataloaders
        self.config = config
        self.save_path = config.save_path
        self.lr_scheduler = INVScheduler(gamma=self.config.gamma,
                                         decay_rate=self.config.decay_rate,
                                         init_lr=self.config.init_lr)
        self.vis = Visualize(env=self.config.exp)
        param_groups = self.model.get_parameter_list()
        self.group_ratios = [group['lr'] for group in param_groups]

    def pretrain_src(self, pretrain_epoch=10):
        self.pprint('Pretrain...')
        kwargs = {'num_workers': 0, 'pin_memory': True}
        val_ratio = .2
        self.train_source_loader = data_loader.load_training(
            self.config.root_path, self.config.source_dir, self.config.batch_size, kwargs, train_val_split=val_ratio)
        self.train_target_loader = data_loader.load_training(
            self.config.root_path, self.config.test_dir, self.config.batch_size, kwargs, train_val_split=val_ratio)
        self.train_source_loader_train, self.train_source_loader_val = self.train_source_loader
        self.train_target_loader_train, self.train_target_loader_val = self.train_target_loader
        best_acc = 0
        self.optimizer_m, _ = self.lr_scheduler.next_optimizer(
            self.group_ratios, self.optimizer_m, 1)

        for epoch in range(pretrain_epoch):
            self.model.c_net.train()
            for _, (x_src, y_src) in enumerate(self.train_source_loader_train):
                x_src, y_src = x_src.cuda(), y_src.cuda()
                inputs = torch.cat([x_src, x_src], dim=0)
                loss_cls, _, _, _, _ = self.model.get_loss(inputs, y_src)
                self.optimizer_m.zero_grad()
                loss_cls.backward()
                self.optimizer_m.step()
            ret = self.evaluate(self.model.c_net, self.train_source_loader_val)
            acc = ret['f1'] if self.config.dataset == 'Covid-19' else ret['accuracy']
            ret = self.evaluate(self.model.c_net, self.test_target_loader)
            acc_tar = ret['f1'] if self.config.dataset == 'Covid-19' else ret['accuracy']
            self.pprint('Epoch: {:2d}, acc src: {:.4f}, acc tar: {:.4f}'.format(
                epoch, acc, acc_tar))
            if best_acc < acc:
                best_acc = acc
                torch.save(self.model.c_net.state_dict(),
                           self.save_path.replace('.mdl', '-pre.mdl'))
        self.pprint('Best pretrain model saved!')

    def train_adapt(self, pretrain_epoch=100):
        self.pprint('Pretrain...')
        kwargs = {'num_workers': 0, 'pin_memory': True}
        train_split = .2
        self.train_source_loader = data_loader.load_training(
            self.config.root_path, self.config.source_dir, self.config.batch_size, kwargs, train_val_split=train_split)
        self.train_target_loader = data_loader.load_training(
            self.config.root_path, self.config.test_dir, self.config.batch_size, kwargs, train_val_split=train_split)
        self.train_source_loader_train, self.train_source_loader_val = self.train_source_loader
        self.train_target_loader_train, self.train_target_loader_val = self.train_target_loader
        best_acc = 0
        for epoch in range(pretrain_epoch):
            losses = []
            self.optimizer_m, lr = self.lr_scheduler.next_optimizer(
                self.group_ratios, self.optimizer_m, 1)
            self.model.c_net.train()
            for datas, datat in zip(self.train_source_loader_train, self.train_target_loader_train):
                (x_src, y_src), (x_tar, _) = datas, datat
                x_src, y_src = x_src.cuda(), y_src.cuda()
                x_tar = x_tar.cuda()
                inputs = torch.cat([x_src, x_tar], dim=0)
                loss_cls, cond_loss, _, _, _ = self.model.get_loss(
                    inputs, y_src)
                self.optimizer_m.zero_grad()
                total_loss = loss_cls + cond_loss
                total_loss.backward()
                self.optimizer_m.step()
                losses.append([loss_cls.item(), cond_loss.item()])
            losses = np.array(losses).mean(0)
            ret = self.evaluate(self.model.c_net, self.train_source_loader_val)
            acc = ret['f1'] if self.config.dataset == 'Covid-19' else ret['accuracy']
            ret = self.evaluate(self.model.c_net, self.test_target_loader)
            acc_tar = ret['f1'] if self.config.dataset == 'Covid-19' else ret['accuracy']
            self.pprint('Epoch: {:2d}, acc src: {:.4f}, acc tar: {:.4f}, lr: {:.4f}'.format(
                epoch, acc, acc_tar, lr))
            print(losses)
            if best_acc < acc:
                best_acc = acc
                torch.save(self.model.c_net.state_dict(),
                           self.save_path.replace('.mdl', '-pre.mdl'))
        self.pprint('Best pretrain model saved!')

    def train(self):
        if self.config.pretrain:
            self.pretrain_src()
        self.model.c_net.load_state_dict(torch.load(
            self.save_path.replace('.mdl', '-pre.mdl')))
        ret = self.evaluate(self.model.c_net, self.test_target_loader)
        acc = ret['accuracy']
        self.pprint('Pretrain acc: {:.4f}'.format(acc))
        self.pprint('Start train...')
        self.train_source_loader_train, self.train_source_loader_val = self.train_source_loader
        self.train_target_loader_train, self.train_target_loader_val = self.train_target_loader
        self.optimizer_g = torch.optim.SGD(
            self.gnet.parameters(), lr=self.config.glr, momentum=.9, weight_decay=self.config.weight_decay)
        iter_num = 0
        epoch = 0
        stop = 0
        mxacc = 0
        lst_cls_loss, lst_gloss, lst_meta_loss = [], [], []
        batch_val = gen_batch(self.train_source_loader_val,
                              self.train_target_loader_val)
        while iter_num < self.max_iter:
            self.model_old.c_net.load_state_dict(self.model.c_net.state_dict())
            loss_train = torch.tensor(0, device='cuda')
            loss_aug = torch.tensor(0, device='cuda')
            self.model.c_net.train()
            self.model_old.c_net.train()
            self.gnet.train()
            self.optimizer_m, lr = self.lr_scheduler.next_optimizer(
                self.group_ratios, self.optimizer_m, iter_num)
            # Meta train
            batch_src, batch_tar = gen_batch(
                self.train_source_loader_train, self.train_target_loader_train, n_batch=1)
            _, (x_src, y_src) = batch_src
            _, (x_tar, _) = batch_tar
            x_src = torch.tensor(
                x_src, dtype=torch.float32, device='cuda', requires_grad=False)
            x_tar = torch.tensor(
                x_tar, dtype=torch.float32, device='cuda', requires_grad=False)
            y_src = torch.tensor(
                y_src, dtype=torch.long, device='cuda', requires_grad=False)
            inputs = torch.cat([x_src, x_tar], dim=0)
            loss_cls, cond_loss2, mar_loss2, feat2, logits2 = self.model.get_loss(
                inputs, y_src)
            loss_train = loss_train + loss_cls
            feat2 = self.model.match_feat(
                cond_loss2, mar_loss2, feat2, logits2)
            g_loss = self.gnet(feat2[:feat2.size(0) // 2],
                               feat2[feat2.size(0) // 2:]).mean()
            loss_aug = loss_aug + 0.1 * g_loss
            lst_cls_loss.append(loss_cls.item())
            lst_gloss.append(g_loss.item())
            self.vis.plot_line([g_loss.item()], iter_num, title='gloss')

            # Compute gradient
            self.optimizer_m.zero_grad()
            # Get grad for OLD loss
            loss_train.backward(retain_graph=True)
            grad_phi = get_grad(self.model.c_net)
            update_params(self.model_old.c_net, grad_phi, lr)

            # Get grad for NEW loss
            loss_aug.backward(create_graph=True)
            grad_phi = get_grad(self.model.c_net)
            update_params(self.model.c_net, grad_phi, lr)

            self.model.c_net.train()
            self.model_old.c_net.train()

            # Meta test
            loss_meta_all = 0.0
            _, (x_src, y_src) = batch_val[0]
            _, (x_tar, y_tar) = batch_val[1]
            x_src = torch.tensor(
                x_src, dtype=torch.float32, device='cuda', requires_grad=False)
            x_tar = torch.tensor(
                x_tar, dtype=torch.float32, device='cuda', requires_grad=False)
            y_src = torch.tensor(
                y_src, dtype=torch.long, device='cuda', requires_grad=False)
            y_tar = torch.tensor(
                y_tar, dtype=torch.long, device='cuda', requires_grad=False)
            inputs_tmp = torch.cat((x_tar, x_src), dim=0)
            cls_loss1, cond_loss1, mar_loss1, feat1, logits1 = self.model_old.get_loss(
                inputs_tmp, y_tar)
            cls_loss2, cond_loss2, mar_loss2, feat2, logits2 = self.model.get_loss(
                inputs_tmp, y_tar)
            g_loss1 = self.gnet(
                feat1[:feat1.size(0) // 2], feat1[feat1.size(0) // 2:]).mean()
            g_loss2 = self.gnet(
                feat2[:feat2.size(0) // 2], feat2[feat2.size(0) // 2:]).mean()
            self.vis.plot_line([mar_loss1.item(), mar_loss2.item(
            )], iter_num, title='MMD loss', legend=['OLD', 'NEW'])
            self.vis.plot_line([g_loss1.item(), g_loss2.item()],
                               iter_num, title='Aug loss', legend=['OLD', 'NEW'])
            loss_meta = gnet_diff(g_loss1, g_loss2)
            loss_meta_all += loss_meta
            lst_meta_loss.append(loss_meta.item())
            iter_num += 1

            self.optimizer_m.step()
            self.optimizer_g.zero_grad()
            loss_meta_all.backward()
            torch.nn.utils.clip_grad_norm_(self.gnet.parameters(), 5)
            self.optimizer_g.step()
            torch.cuda.empty_cache()

            if iter_num % 30 == 0:
                stop += 1
                calcf1 = True if self.config.dataset == 'Covid-19' else False
                ret = self.evaluate(self.model.c_net, self.test_target_loader)
                acc = ret['accuracy']
                self.vis.plot_line(
                    [acc], epoch, 'Acc on target', legend=['Tar'])
                if acc >= mxacc:
                    stop = 0
                    mxacc = acc
                    torch.save(self.model.c_net.state_dict(), self.save_path)
                if not calcf1:
                    self.pprint('\nEpoch:[{:.0f}({:.2f}%)], cls_loss: {:.5f}, g_loss: {:.6f}, diff: {:.6f}, acc:{:.4f}, mxacc:{:.4f}'.format(
                        epoch, float(iter_num) * 100. / self.max_iter, np.array(lst_cls_loss).mean(), np.array(lst_gloss).mean(), np.array(lst_meta_loss).mean(), acc, mxacc))
                    self.vis.plot_line([np.array(lst_cls_loss).mean(), np.array(lst_gloss).mean(), np.array(
                        lst_meta_loss).mean()], epoch, title='Loss', legend=['cls', 'gnet', 'val'])
                else:
                    self.pprint('\nEpoch:[{:.0f}/({:.2f}%)], acc: {:.4f}, mxacc: {:.4f}'.format(
                        epoch, float(iter_num) * 100. / self.max_iter, ret['accuracy'], mxacc))
                    self.pprint(ret['metr'])
                epoch += 1
            lst_cls_loss, lst_gloss, lst_meta_loss = [], [], []

            # Reload data every 10 epochs
            if epoch > 0 and epoch % 2 == 0:
                np.random.seed(self.config.seed + 1)
                kwargs = {'num_workers': 1, 'pin_memory': True}
                val_split = .1
                self.train_source_loader = data_loader.load_training(
                    self.config.root_path, self.config.source_dir, self.config.batch_size, kwargs, train_val_split=val_split)
                self.train_target_loader = data_loader.load_training(
                    self.config.root_path, self.config.test_dir, self.config.batch_size, kwargs, train_val_split=val_split)
                batch_val = gen_batch(
                    self.train_source_loader_val, self.train_target_loader_val)

            if stop >= self.config.early_stop:
                self.pprint('=================Early stop!!')
                break

        self.pprint('Max result: ' + str(mxacc))

    def evaluate(self, model, input_loader):
        model.eval()
        all_probs, all_labels = None, None
        with torch.no_grad():
            for inputs, labels in input_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                if self.config.multi_gpu:
                    probs, _ = model.module.predict(inputs)
                else:
                    probs, _ = model.predict(inputs)
                labels = labels.data.long()
                if all_probs is None:
                    all_probs = probs
                    all_labels = labels
                else:
                    all_probs = torch.cat((all_probs, probs), 0)
                    all_labels = torch.cat((all_labels, labels), 0)
        probs, predict = torch.max(all_probs, 1)
        res = metric(all_labels.cpu().detach().numpy(), predict.cpu(
        ).detach().numpy(), probs.cpu().detach().numpy())
        return res

    def pprint(self, *text):
        # print with UTC+8 time
        log_file = self.config.save_path.replace('.mdl', '.log')
        curr_time = '['+str(datetime.datetime.utcnow() +
                            datetime.timedelta(hours=8))[:19]+'] -'
        print(curr_time, *text, flush=True)
        if not os.path.exists(log_file):
            if not os.path.exists(self.config.save_folder):
                os.mkdir(self.config.save_folder)
            fp = open(self.config.log_file, 'w')
            fp.close()
        with open(log_file, 'a') as f:
            print(curr_time, *text, flush=True, file=f)


def get_grad(model):
    grad_phi = []
    for _, (k, v) in enumerate(model.state_dict().items()):
        if k.__contains__('bn'):
            grad_phi.append(None)
        else:
            grad_phi.append(v.grad)
    return grad_phi


def update_grad(model, grad_phi, lr):
    num_grad = 0
    phi_updated_new = {}
    for _, (k, v) in enumerate(model.state_dict().items()):
        if grad_phi[num_grad] is None:
            num_grad += 1
            phi_updated_new[k] = v
        else:
            phi_updated_new[k] = v - \
                lr * grad_phi[num_grad]
            num_grad += 1
    return phi_updated_new


class INVScheduler(object):
    def __init__(self, gamma, decay_rate, init_lr=0.001):
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.init_lr = init_lr

    # def next_optimizer(self, group_ratios, optimizer, epoch, nepoch):
    #     lr = self.init_lr * math.pow((1 + 10 * float(epoch) / nepoch), -0.75)
    #     # lr = self.init_lr * (1 + self.gamma * num_iter) ** (-self.decay_rate)
    #     i = 0
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr * group_ratios[i]
    #         i += 1
    #     return optimizer, lr

    def next_optimizer(self, group_ratios, optimizer, num_iter):
        # lr = self.init_lr * math.pow((1 + 10 * float(epoch) / nepoch), -0.75)
        lr = self.init_lr * (1 + self.gamma * num_iter) ** (-self.decay_rate)
        i = 0
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * group_ratios[i]
            i += 1
        return optimizer, lr


def gen_batch(src_loader, tar_loader, n_batch=1):
    src_list, tar_list = list(
        enumerate(src_loader)), list(enumerate(tar_loader))
    len_src, len_tar = len(src_list), len(tar_list)
    idx_src, idx_tar = np.random.randint(len_src), np.random.randint(len_tar)
    return src_list[idx_src], tar_list[idx_tar]


if __name__ == "__main__":
    a = L2MTrainerCritic(None, None, None, None, None, None, None, None)
    print(a)
