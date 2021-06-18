import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Implement monotonic networks. Reference:

Sill J. Monotonic networks[C]//Advances in neural information processing systems. 1998: 661-667.
'''
class PositiveLinear(nn.Linear):
    def forward(self, input):
        return F.linear(input, self.weight**2, self.bias)


"""A MLP (Multi-layer perceptron) class that can represent:
        - Vanilla MLP (by default)
        - Monotonic networks (by setting mono=True)
"""
class MLP(nn.Module):

    def __init__(self, n_input, n_hiddens, n_output, drop_out=0, mono=False, init_net=True):
        """Init func

        Args:
            n_input (int): input dim
            n_hiddens (list): a list of all hidden units
            n_output (int): output dim
            drop_out (float, optional): drop_out rate. Defaults to 0.
            mono (bool, optional): Monotonic networks or not. Defaults to False.
            init_net (bool, optional): Initialize the network or not. Defaults to True.
        """
        super(MLP, self).__init__()
        self.is_mono = mono
        self.drop_out = drop_out
        hidden = []
        for layer in n_hiddens:
            if self.is_mono:
                hidden.append(PositiveLinear(n_input, layer))
            else:
                hidden.append(nn.Linear(n_input, layer))
            hidden.append(nn.ReLU())
            if self.drop_out != 0:
                hidden.append(nn.Dropout(self.drop_out))
            n_input = layer
        self.hidden = nn.Sequential(*hidden)
        if self.is_mono:
            self.out = PositiveLinear(n_input, n_output)
        else:
            self.out = nn.Linear(n_input, n_output)
        if init_net:
            self.init_layers()

    def forward(self, x):
        x = self.hidden(x)
        x = self.out(x)
        return x

    def init_layers(self, weight=0.01, bias=0):
        """Initialize the layers

        Args:
            weight (float, optional): weight. Defaults to 0.01.
            bias (int, optional): bias. Defaults to 0.
        """
        for layer in self.hidden:
            if self.is_mono:
                if isinstance(layer, PositiveLinear):
                    layer.weight.data.normal_(0, weight)
                    layer.bias.data.fill_(bias)
            else:
                if isinstance(layer, nn.Linear):
                    layer.weight.data.normal_(0, weight)
                    layer.bias.data.fill_(bias)
        self.out.weight.data.normal_(0, weight)
        self.out.bias.data.fill_(bias)


if __name__ == '__main__':
    mlp = MLP(10, [10, 20, 30], 40)
    print(mlp)