'''
Gradient reversal layer. Reference:
Ganin et al. Unsupervised domain adaptation by backpropagation. ICML 2015.
'''

import torch
import torch.nn as nn

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

class LambdaSheduler(nn.Module):
    def __init__(self, gamma=1.0, max_iter=1000, **kwargs):
        super(LambdaSheduler, self).__init__()
        self.gamma = gamma
        self.max_iter = max_iter
        self.curr_iter = 0

    def lamb(self):
        p = self.curr_iter / self.max_iter
        lamb = 2. / (1. + np.exp(-self.gamma * p)) - 1
        return lamb
    
    def step(self):
        self.curr_iter = min(self.curr_iter + 1, self.max_iter)

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class AdversarialLoss(LambdaSheduler):
    def __init__(self, gamma=1.0, max_iter=1000, domain_classifier=None, **kwargs):
        super(AdversarialLoss, self).__init__(gamma=gamma, max_iter=max_iter, **kwargs)
        if domain_classifier:
            self.domain_classifier = domain_classifier
        else:
            self.domain_classifier = Discriminator()
    
    def forward(self, source, target):
        lamb = self.lamb()
        self.step()
        source_loss = self.get_adversarial_result(source, True, lamb)
        target_loss = self.get_adversarial_result(target, False, lamb)
        adv_loss = 0.5 * (source_loss + target_loss)
        return adv_loss
    
    def get_adversarial_result(self, x, source=True, lamb=1.0):
        x = ReverseLayerF.apply(x, lamb)
        domain_pred = self.domain_classifier(x)
        device = domain_pred.device
        if source:
            domain_label = torch.ones(len(x), 1).long()
        else:
            domain_label = torch.zeros(len(x), 1).long()
        loss_fn = nn.BCELoss()
        loss_adv = loss_fn(domain_pred, domain_label.float().to(device))
        return loss_adv

class Discriminator(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ]
        self.layers = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)