import torch
import torch.nn as nn
from . import MLP
from utils.pdist import pdist

# GNet: version 3, using gram matrix
class GNetGram(nn.Module):
    def __init__(self, n_input, n_hiddens, n_output, use_set=True, drop_out=0, mono=False, init_net=True):
        """Init func for GNet
        TODO: Gram matrix -> vector
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
        out = torch.sigmoid(out)
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


# GNet: version 2, using set and difference
class GNet2(nn.Module):
    def __init__(self, n_input, n_hiddens, n_output, use_set=True, drop_out=0, mono=False):
        """Init func for GNet

        Args:
            n_input (int): input dim
            n_hiddens (list): a list of hidden dims
            n_output (int): output dim
            use_set (bool, optional): Deep set or not. Defaults to True.
            drop_out (float, optional): drop out. Defaults to 0.
            mono (bool, optional): Monotonic network or not. Defaults to False.
        """
        super(GNet2, self).__init__()
        self.use_set = use_set
        self.net_s = MLP(n_input, n_hiddens, n_output,
                         drop_out=drop_out, mono=mono)
        self.net_t = MLP(n_input, n_hiddens, n_output,
                         drop_out=drop_out, mono=mono)
        self.combine = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_out),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_out),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_out),
            nn.Linear(256, 1)
        )
        self.parameter_list = [{"params": self.combine.parameters(), "lr": 1}]

    def forward(self, x):
        x_src, x_tar = x[:x.size(0) // 2], x[x.size(0) // 2 : ]
        # fea_src, fea_tar = F.relu(self.net_s(x_src)), F.relu(self.net_s(x_tar))
        
        fea_src, fea_tar = x_src, x_tar
        f_src, f_tar = None, None
        if self.use_set:
            f_src = torch.mean(fea_src, dim=0, keepdim=True)
            f_tar = torch.mean(fea_tar, dim=0, keepdim=True)
        feas = f_src - f_tar
        out = self.combine(feas)
        # out = out * out.t()
        # out = F.relu(out)
        # out = torch.sigmoid(out)
        return out

    def get_parameter_list(self):
        if isinstance(self, torch.nn.DataParallel) or isinstance(self, torch.nn.parallel.DistributedDataParallel):
            return self.module.parameter_list
        return self.parameter_list


