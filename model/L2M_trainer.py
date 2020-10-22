import torch
import numpy as np
import copy
import datetime
import os
from utils.utils_f1 import metric
import copy


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


def gnet_diff(g_loss_pre, g_loss_post):
    """Difference between previous and current gnet losses: tanh(g_loss_pre, g_loss_post)
    May use different functions such as subtraction, log, or crossentropy...

    Args:
        g_loss_pre (flaot): previous gnet loss
        g_loss_post (float): current gnet loss

    Returns:
        diff (float): difference
    """
    # diff = 100 * (g_loss_pre - g_loss_post)
    diff = torch.tanh(g_loss_post - g_loss_pre)
    # diff = torch.tanh(torch.nn.CrossEntropyLoss()(g_loss_pre, g_loss_post))
    # diff = torch.log(g_loss_pre) - torch.log(g_loss_post)
    return diff


class L2MTrainer(object):
    """Main training class
    """

    def __init__(self, gnet, model, model_old, optimizer_m, optimizer_g, dataloaders, config, max_iter=200000) -> None:
        """Init func

        Args:
            gnet (GNet): gnet
            model (model): model
            model_old (old model): old model
            optimizer_m (nn.optim): optimizer for optimizing L2M model
            optimizer_g (nn.optim): optimizer for optimizing GNet
            dataloaders (list): [source_train_loader, target_train_loader, target_test_loader]
            config (dict): config dict
            max_iter (int, optional): maximum iteration num. Defaults to 200000.
        """
        self.gnet = gnet
        self.model = model
        self.model_old = model_old
        self.optimizer_m = optimizer_m
        self.optimizer_g = optimizer_g
        self.max_iter = max_iter
        self.train_source_loader, self.train_target_loader, self.test_target_loader = dataloaders
        self.config = config
        self.save_path = config.save_path
        self.glr = config.glr
        self.lr_scheduler = INVScheduler(gamma=self.config.gamma,
                                         decay_rate=self.config.decay_rate,
                                         init_lr=self.config.init_lr)

    def train(self):
        self.pprint('Start train...')
        iter_num = 0
        epoch = 0
        stop = 0
        mxacc = 0
        param_groups = self.model.get_parameter_list()
        group_ratios = [group['lr'] for group in param_groups]
        while True:
            vlr = self.glr * ((0.1 ** int(epoch >= 300))
                              * (0.1 ** int(epoch >= 600)))
            self.model_old.c_net.load_state_dict(self.model.c_net.state_dict())
            self.gnet.train()
            self.model_old.c_net.train()
            self.model.c_net.train()
            self.optimizer_g = torch.optim.Adam(
                self.gnet.parameters(), lr=vlr)

            # construct meta_loader from target_loader before each epoch
            if epoch == 0:
                meta_loader = generate_metadata(
                    m=5, loader=self.test_target_loader)
            else:
                meta_loader = generate_metadata_soft(
                    m=5, train_target_loader=self.test_target_loader, model=self.model)

            iter_meta = iter(meta_loader)
            lst_cls, lst_val = [], []
            for (datas, datat) in zip(self.train_source_loader, self.train_target_loader):
                inputs_source, labels_source = datas
                inputs_target, _ = datat
                if inputs_source.size(0) != inputs_target.size(0):
                    continue
                try:
                    meta_data, meta_label = iter_meta.next()
                except:
                    iter_meta = iter(meta_loader)
                    meta_data, meta_label = iter_meta.next()
                meta_data, meta_label = meta_data.cuda(), meta_label.cuda()

                self.optimizer_m, lr = self.lr_scheduler.next_optimizer(
                    group_ratios, self.optimizer_m, iter_num/5)
                inputs_source, inputs_target, labels_source = inputs_source.cuda(
                ), inputs_target.cuda(), labels_source.cuda()

                inputs = torch.cat((inputs_source, inputs_target), dim=0)
                set_require_grad(self.model.c_net, True)
                classifier_loss, cond_loss, mar_loss, feat, logits = self.model.get_loss(
                    inputs, labels_source)
                m_feat = self.model.match_feat(
                    cond_loss, mar_loss, feat, logits)
                with torch.no_grad():
                    mv_lambda = self.gnet(m_feat.detach().data)
                loss = mv_lambda.mean()
                total_loss = classifier_loss + loss
                self.optimizer_m.zero_grad()
                total_loss.backward()
                self.optimizer_m.step()

                # diff_g = self.update_theta(inputs_source, meta_data, labels_source, total_loss, classifier_loss, laux)
                inputs_tmp = torch.cat((inputs_source, meta_data), dim=0)
                _, cond_loss, mar_loss, feat, logits = self.model_old.get_loss(
                    inputs_tmp, labels_source)
                m_feat = self.model_old.match_feat(
                    cond_loss, mar_loss, feat, logits)
                g_loss_pre = self.gnet(m_feat).mean()

                _, cond_loss2, mar_loss2, feat2, logits2 = self.model.get_loss(
                    inputs_tmp, labels_source)
                m_feat2 = self.model.match_feat(
                    cond_loss2, mar_loss2, feat2, logits2)
                g_loss_post = self.gnet(m_feat2).mean()

                diff = gnet_diff(g_loss_pre, g_loss_post)
                self.optimizer_g.zero_grad()
                diff.backward()
                self.optimizer_g.step()
                # diff_g = torch.tensor(0)

                lst_cls.append(classifier_loss.item())
                lst_val.append(diff.item())
                iter_num += 1
            # print(self.model.c_net.classifier_layer[0].weight.grad.sum())
            # print(self.gnet.out.weight.grad.sum())
            stop += 1
            calcf1 = True if self.config.dataset == 'Covid-19' else False
            ret = evaluate(self.model, self.test_target_loader, calcf1=calcf1)
            acc = ret['metr']['f1'] if self.config.dataset == 'Covid-19' else ret['accuracy']
            if acc >= mxacc:
                stop = 0
                mxacc = acc
                torch.save(self.model.c_net.state_dict(), self.save_path)
            if not calcf1:
                classifier_loss = np.array(lst_cls).mean()
                gnet_loss = np.array(lst_val).mean()
                self.pprint('\nEpoch:[{:.0f}({:.2f}%)], cls_loss: {:.5f}, g_loss: {:.10f}, acc:{:.4f}, mxacc:{:.4f}'.format(
                    epoch, float(iter_num) * 100. / self.max_iter, classifier_loss, gnet_loss, acc, mxacc))
            else:
                self.pprint('\nEpoch:[{:.0f}/({:.2f}%)], acc: {:.4f}, mxacc: {:.4f}'.format(
                    epoch, float(iter_num) * 100. / self.max_iter, ret['accuracy'], mxacc))
                self.pprint(ret['metr'])
            epoch += 1

            if iter_num >= self.max_iter:
                self.pprint('finish train')
                break
            if stop >= self.config.early_stop:
                self.pprint('=================Early stop!!')
                break
        self.pprint('Max result: ' + str(mxacc))
        self.pprint('Train is finished!')

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


class INVScheduler(object):
    def __init__(self, gamma, decay_rate, init_lr=0.001):
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.init_lr = init_lr

    def next_optimizer(self, group_ratios, optimizer, num_iter):
        lr = self.init_lr * (1 + self.gamma * num_iter) ** (-self.decay_rate)
        i = 0
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * group_ratios[i]
            i += 1
        return optimizer, lr


def get_softlabel(model, loader):
    model.c_net.eval()
    first_test = True
    all_probs = None
    for datat in loader:
        inputs_t, _ = datat
        inputs_t = inputs_t.cuda()
        probabilities = model.predict(inputs_t)
        probabilities = probabilities.data.float()
        if first_test:
            all_probs = probabilities
            first_test = False
        else:
            all_probs = torch.cat((all_probs, probabilities), 0)
    prob, predict = torch.max(all_probs, 1)
    model.c_net.train()
    return prob, predict


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
        if c != len(start_idx)-1:
            shuffleimgs = targetimgs[start_idx[c]:start_idx[c+1]]
        else:
            shuffleimgs = targetimgs[start_idx[c]:]
        np.random.shuffle(shuffleimgs)
        metaimgs.extend(shuffleimgs[:m])
    meta_loader = copy.deepcopy(loader)
    meta_loader.dataset.imgs = metaimgs
    return meta_loader


def generate_metadata_soft(m, train_target_loader, model):
    prob, softlabels = get_softlabel(model, train_target_loader)
    test_imgs = train_target_loader.dataset.imgs
    mask = prob >= 0.8
    # confident_imgs, confident_labels = test_imgs[mask], softlabels[mask]
    newimgs = []
    for i in range(len(train_target_loader.dataset.imgs)):
        if i < len(softlabels):
            img = tuple([train_target_loader.dataset.imgs[i]
                         [0], softlabels[i].item()])
            newimgs.append(img)
        else:
            newimgs.append(train_target_loader.dataset.imgs[i])
    lbs = []
    for _, i in newimgs:
        lbs.append(i)
    lbs = np.array(lbs)
    metaimgs = []
    metaimgs_idx = []
    for c in range(len(train_target_loader.dataset.classes)):
        cindx = np.where(lbs == c)[0]
        if len(cindx) >= m:
            np.random.shuffle(cindx)
            metaimgs_idx.extend(list(np.random.choice(cindx, m)))
        elif len(cindx) != 0:
            metaimgs_idx.extend(list(cindx))
        else:
            continue
    for i in metaimgs_idx:
        metaimgs.append(newimgs[i])
    meta_loader = copy.deepcopy(train_target_loader)
    meta_loader.dataset.imgs = metaimgs
    return meta_loader


def set_require_grad(model, grad=True):
    for item in model.parameters():
        item.requires_grad = grad


def evaluate(model, input_loader, calcf1=False):
    model.c_net.eval()
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True
    all_probs, all_labels = None, None
    for _ in range(num_iter):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1]
        inputs = inputs.cuda()
        labels = labels.cuda()
        probabilities = model.predict(inputs)
        probabilities = probabilities.data.float()
        labels = labels.data.float()
        if first_test:
            all_probs = probabilities
            all_labels = labels
            first_test = False
        else:
            all_probs = torch.cat((all_probs, probabilities), 0)
            all_labels = torch.cat((all_labels, labels), 0)
    ret = {}
    probs, predict = torch.max(all_probs, 1)
    if calcf1:
        metr = metric(all_labels.cpu().detach().numpy(), predict.cpu(
        ).detach().numpy(), probs.cpu().detach().numpy())
        ret['metr'] = metr
    accuracy = torch.sum(torch.squeeze(predict).float() ==
                         all_labels).item() / float(all_labels.size()[0])
    model.c_net.train()
    ret['accuracy'] = accuracy
    return ret


if __name__ == "__main__":
    a = L2MTrainer(None, None, None, None, None, None, None, None)
    print(a)
