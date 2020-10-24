import torch
import torch.nn as nn
from . import MLP

'''
GNet, version 1
'''
class GNet(nn.Module):
    def __init__(self, n_input, n_hiddens, n_output, use_set=True, drop_out=0, mono=False):
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
        super(GNet, self).__init__()
        self.net = MLP(1024, n_hiddens, n_output, drop_out, mono=mono)
        # self.hidden = net.hidden
        # self.out = net.out
        self.out = nn.Linear(1024, n_output)
        self.set = use_set
        self.parameter_list = [{"params": self.out.parameters(), "lr": 0.1}]

    def forward(self, x):
        # x = self.hidden(x)
        gram = pdist(x[:x.size(0) // 2], x[x.size(0) // 2:])
        flatten = gram.view(-1)
        x = self.net(flatten)
        # if self.set:
        #     x_s = torch.mean(x[:x.size(0) // 2], dim=0, keepdim=True)
        #     x_t = torch.mean(x[x.size(0) // 2:], dim=0, keepdim=True)
        #     x = x_s - x_t
        # x = self.out(x)
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


def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    r"""Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)