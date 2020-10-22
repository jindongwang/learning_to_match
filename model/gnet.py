import torch
import torch.nn as nn
from . import MLP

'''
GNet, version 1
'''
class GNet(nn.Module):
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
        super(GNet, self).__init__()
        net = MLP(n_input, n_hiddens, n_output, drop_out, mono=mono)
        self.hidden = net.hidden
        self.out = net.out
        self.set = use_set
        self.parameter_list = [{"params": self.hidden.parameters(), "lr": 0.1}, {
            "params": self.out.parameters(), "lr": 0.1}]

    def forward(self, x):
        x = self.hidden(x)
        if self.set:
            x_s = torch.sum(x[:x.size(0) // 2], dim=0, keepdim=True)
            x_t = torch.sum(x[x.size(0) // 2:], dim=0, keepdim=True)
            x = x_s - x_t
        x = self.out(x)
        out = torch.sigmoid(x)
        return out


'''
GNet, version 2
'''
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

    def forward(self, x_src, x_tar):
        # fea_src, fea_tar = F.relu(self.net_s(x_src)), F.relu(self.net_s(x_tar))
        fea_src, fea_tar = x_src, x_tar
        if self.use_set:
            fea_src = torch.mean(fea_src, dim=0, keepdim=True)
            fea_tar = torch.mean(fea_tar, dim=0, keepdim=True)
        feas = fea_src - fea_tar
        out = self.combine(feas)
        # out = out * out.t()
        # out = F.relu(out)
        # out = torch.sigmoid(out)
        return out
