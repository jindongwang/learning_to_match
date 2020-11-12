import torch
import torch.nn as nn
from . import MLP
from utils.pdist import pdist

# GNet: using gram matrix
class GNetGram(nn.Module):
    def __init__(self, n_input, n_hiddens, n_output, use_set=True, drop_out=0, mono=False, init_net=True):
        """Init func for GNet with Gram matrix
        Args:
            n_input (int): input dim
            n_hiddens (list): a list of hidden dims
            n_output (int): output dim
            use_set (bool, optional): Deep set or not. Defaults to True.
            drop_out (float, optional): drop out. Defaults to 0.
            mono (bool, optional): Monotonic network or not. Defaults to False.
        """
        super(GNetGram, self).__init__()
        self.net = MLP(n_input, n_hiddens, n_output, drop_out, mono=mono, init_net=init_net)
        self.set = use_set
        self.parameter_list = [{"params": self.net.parameters(), "lr": 1}]
        

    def forward(self, x):
        gram = pdist(x[:x.size(0) // 2], x[x.size(0) // 2:])
        flatten = gram.view(-1)
        out = self.net(flatten)
        out = torch.tanh(out)
        # out = torch.sigmoid(out)
        # out = torch.nn.Softplus()(out)
        # out = torch.nn.ReLU()(out)
        return out

    def forward2(self, x, y):
        xs, ys = x[:x.size(0) // 2], y[:y.size(0) // 2]
        xt, yt = x[x.size(0) // 2:], y[y.size(0) // 2:]
        gram = torch.ones(1).cuda()
        label_set = torch.unique(y)
        for c in label_set:
            ind = ys == c
            xs_c, ys_c = xs[ind], ys[ind]
            ind = yt == c
            xt_c, yt_c = xt[ind], yt[ind]
            gram_c = pdist(xs_c, xt_c).view(-1)
            gram = torch.cat((gram, gram_c))
        out = self.net(gram[1:])
        out = torch.tanh(out)
        return out

    def get_parameter_list(self):
        if isinstance(self, torch.nn.DataParallel) or isinstance(self, torch.nn.parallel.DistributedDataParallel):
            return self.module.parameter_list
        return self.parameter_list

# Gnet: version 1, just an MLP
class GNet(nn.Module):
    def __init__(self, n_input, n_hiddens, n_output, use_set=True, drop_out=0, mono=False, init_net=True):
        """Init func for GNet
        Args:
            n_input (int): input dim
            n_hiddens (list): a list of hidden dims
            n_output (int): output dim
            use_set (bool, optional): Deep set or not. Defaults to True.
            drop_out (float, optional): drop out. Defaults to 0.
            mono (bool, optional): Monotonic network or not. Defaults to False.
        """
        super(GNet, self).__init__()
        net = MLP(n_input, n_hiddens, n_output, drop_out, mono=mono, init_net=init_net)
        self.hidden = net.hidden
        self.out = net.out
        self.set = use_set
        self.parameter_list = [{"params": self.out.parameters(), "lr": 1}, {"params": self.hidden.parameters(), "lr": 1}]

    def forward(self, x):
        if self.set:
            x_s = torch.mean(x[:x.size(0) // 2], dim=0, keepdim=True)
            x_t = torch.mean(x[x.size(0) // 2:], dim=0, keepdim=True)
            x = x_s - x_t
        x_hidden = self.hidden(x)
        out = self.out(x_hidden)
        out = torch.sigmoid(x)
        return out


    def get_parameter_list(self):
        if isinstance(self, torch.nn.DataParallel) or isinstance(self, torch.nn.parallel.DistributedDataParallel):
            return self.module.parameter_list
        return self.parameter_list

