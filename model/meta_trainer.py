import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy
import numpy as np
import argparse

import model.backbone as backbone
import torch.nn.functional as F
import torch

import data_loader
import pretty_errors

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='ERM')
    parser.add_argument('--root_path', type=str, default="/data/jindwang/OfficeHome",
                        help='the path to load the data')
    parser.add_argument('--source_dir', type=str, default="Art",
                        help='the name of the source dir')
    parser.add_argument('--test_dir', type=str, default="Clipart",
                        help='the name of the test dir')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_class', type=int, default=65)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=.9)
    parser.add_argument('--N_EPOCH', type=int, default=200)
    parser.add_argument('--early_stop', type=int, default=30)
    args = parser.parse_args()
    return args

def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


class ERM(nn.Module):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, base_net='ResNet50', bottleneck_dim=2048, width=2048, class_num=65):
        super(ERM, self).__init__()
        self.base_network = backbone.network_dict[base_net]()
        # self.bottleneck_layer = nn.Sequential(*[nn.Linear(self.base_network.output_num(), bottleneck_dim),
        #                                         nn.BatchNorm1d(bottleneck_dim),
        #                                         nn.ReLU(),
        #                                         nn.Dropout(0.5, inplace=False)])
        self.classifier_layer = nn.Sequential(*[
            # nn.Linear(bottleneck_dim, width),
            #                                     nn.ReLU(),
            #                                     nn.Dropout(0.5),
                                                nn.Linear(2048, class_num)])
        

        self.network = nn.Sequential(self.base_network, self.classifier_layer)
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum
        )


    
    def fp(self, x):
        f = self.base_network(x)
        out_logits = self.classifier_layer(f)
        return out_logits

    def update(self, minibatches, unlabeled=None):
        all_x = minibatches[0][0]
        all_y = minibatches[0][1]
        pred = self.fp(all_x)
        loss = F.cross_entropy(pred, all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.fp(x)


class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """
    def __init__(self, base_net='ResNet50', bottleneck_dim=2048, width=2048, class_num=65, hparams=None):
        super(Mixup, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        objective = 0
        ypred = torch.max(self.network(minibatches[1][0]), 1)[1]
        minibatches = [minibatches[0], (minibatches[0][0], ypred)]
        
        for (xi, yi), (xj, yj) in minibatches:
            lam = np.random.beta(self.hparams["mixup_alpha"],
                                 self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.fp(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item()}

class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """
    def __init__(self, base_net='ResNet50', bottleneck_dim=2048, width=2048, class_num=65, hparams=None):
        super(GroupDRO, self).__init__()
        self.hparams = hparams
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches, unlabeled=None):

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).cuda()

        losses = torch.zeros(len(minibatches)).cuda()

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """
    def __init__(self, base_net='ResNet50', bottleneck_dim=2048, width=2048, class_num=65, hparams=None):
        super(MLDG, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)

            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)

            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)

        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        ypred = torch.max(self.network(minibatches[1][0]), 1)[1]
        minibatches = [minibatches[0], (minibatches[0][0], ypred)]
        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(self.network)

            inner_opt = torch.optim.Adam(
                inner_net.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay
            )

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(),
                                    inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(),
                allow_unused=True)

            # `objective` is populated for reporting purposes
            objective += (self.hparams['mldg_beta'] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(
                        self.hparams['mldg_beta'] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        self.optimizer.step()

        return {'loss': objective}


class MEDA(ERM):
    """
    Deep version of MEDA
    """
    def __init__(self, base_net='ResNet50', bottleneck_dim=2048, width=2048, class_num=65, hparams=None):
        super(Mixup, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        objective = 0
        ypred = torch.max(self.network(minibatches[1][0]), 1)[1]
        minibatches = [minibatches[0], (minibatches[0][0], ypred)]
        
        for (xi, yi), (xj, yj) in minibatches:
            lam = np.random.beta(self.hparams["mixup_alpha"],
                                 self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.fp(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item()}

def train(dataloaders, model):
    train_loader, val_loader, test_loader = dataloaders
    stop = 0
    mx = 0
    for e in range(args.N_EPOCH):
        stop += 1
        model.train()
        losses = []
        for (ds, dt) in zip(train_loader, val_loader):
            (xs, ys), (xt, yt) = ds, dt
            xs, ys, xt, yt = xs.cuda(), ys.cuda(), xt.cuda(), yt.cuda()
            minibatch = [(xs, ys), (xt, yt)]
            loss = model.update(minibatch)['loss']
            losses.append(loss)
        loss = np.array(losses).mean()
        # validation
        acc_train = test_model(model, train_loader)
        acc_val = test_model(model, val_loader)
        acc_test = test_model(model, test_loader)
        if acc_val > mx:
            mx = acc_val
            torch.save(model.state_dict(), f'{args.algo}.pkl')
            stop = 0
        print(f'Epoch [{e:2d}/{args.N_EPOCH}]: loss: {loss:.4f}, acc_train: {acc_train:.4f}, acc_val: {acc_val:.4f}, acc_test: {acc_test:.4f}')
        if stop >= args.early_stop:
            print('---early stop---')
            break
    acc_test = test_model(model, test_loader, f'{args.algo}.pkl')
    print(f'Final test result: {acc_test}')

def load_data(root_path, source_dir, target_dir, batch_size):
    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_val_split = -1
    source_loader = data_loader.load_training(
        root_path, source_dir, batch_size, kwargs, train_val_split)
    target_loader = data_loader.load_training(
        root_path, target_dir, batch_size, kwargs, train_val_split)
    test_loader = data_loader.load_testing(
        root_path, target_dir, batch_size, kwargs)
    return source_loader, target_loader, test_loader


def test_model(model, test_loader, model_file=None):
    if model_file is not None:
        model.load_state_dict(torch.load(model_file))

    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for sample, label in test_loader:
            sample, label = sample.cuda().float(), label.cuda().long()
            weighted_output = model.predict(sample)
            _, predicted = torch.max(weighted_output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum()
    acc_test = float(correct) * 100 / total
    return acc_test

if __name__ == '__main__':
    args = get_args()
    print(vars(args))
    train_source_loader, train_target_loader, test_target_loader = load_data(
        args.root_path, args.source_dir, args.test_dir, args.batch_size)
    model = None
    if args.algo.lower() in ['baseline', 'erm']: # baseline
        model = ERM(base_net='ResNet50', bottleneck_dim=1024, width=256, class_num=args.n_class)
    elif args.algo.lower() == 'mixup':
        model = Mixup(base_net='ResNet50', bottleneck_dim=1024, width=256, class_num=args.n_class, hparams={'mixup_alpha': 0.2})
    elif args.algo.lower() == 'groupdro':
        model = GroupDRO(args.n_class, {'groupdro_eta': 1e-2})
    elif args.algo.lower() == 'mldg':
        model = MLDG(base_net='ResNet50', bottleneck_dim=1024, width=256, class_num=args.n_class, hparams={'mldg_beta': 1.})
    assert model is not None, 'Specify models!'
    model.cuda()
    train((train_source_loader, train_target_loader, test_target_loader), model)
    
