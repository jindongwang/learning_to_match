import torch

def entropy(out_prob):
    """ 熵，输入是softmax后的各个类的概率 """
    d = torch.distributions.Categorical(probs=out_prob)
    ent = d.entropy()
    return ent
    