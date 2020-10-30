'''
Gradient reversal layer. Reference:
Ganin et al. Unsupervised domain adaptation by backpropagation. ICML 2015.
'''

import torch

# For pytorch version > 1.0
# Usage:
# b = GradReverse.apply(a, 1) # 1 is the lambda value, you are free to set it
class GradReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

# For pytorch version 1.0
# Usage:
# grl = GradientReverseLayer(1)  # 1 is the lambda value, you are free to set it
# b = grl(a)
class GradientReverseLayer(torch.autograd.Function):
    def __init__(self, lambd=1):
        self.lambd = lambd

    def forward(self, input):
        output = input * 1.0
        return output

    def backward(self, grad_output):
        return -self.lambd * grad_output
