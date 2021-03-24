import numpy as np
import random
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt


def metric(y_true, y_pred, y_score, draw_curve=False):
    prec, rec, f1, auc = 0, 0, 0, 0
    if len(np.unique(y_true)) == 2:
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary')
        # this is to ensure that we only use the probability for y=1 class to draw curves
        for i in range(len(y_true)):
            if y_pred[i] == 0:
                y_score[i] = 1 - y_score[i]
        auc = roc_auc_score(y_true, y_score)
        if draw_curve:
            precision, recall, thresholds = precision_recall_curve(
                y_true, y_score)
            np.savetxt('prec.csv', precision, fmt='%.2f', delimiter=',')
            np.savetxt('recall.csv', recall, fmt='%.2f', delimiter=',')
            plt.plot(precision, recall, marker='o')
            plt.savefig('pr_curve.png')

            plt.clf()
            fpr, tpr, threshold = roc_curve(y_true, y_score, pos_label=1)
            np.savetxt('fpr.csv', fpr, fmt='%.2f', delimiter=',')
            np.savetxt('tpr.csv', tpr, fmt='%.2f', delimiter=',')
            plt.plot(fpr, tpr, marker='o')
            plt.savefig('roc_curve.png')
    acc = accuracy_score(y_true, y_pred)
    metr = {'f1': f1, 'r': rec, 'p': prec, 'auc': auc, 'accuracy': acc}
    return metr

# controls random seed
def set_random(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}: {val' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)



def mdd_loss(outputs, len_src, cls_adv, srcweight):
    """Implement the MDD loss. Reference:
    Zhang et al. Bridging theory and algorithm for domain adpatation. ICML 2019.

    Args:
        outputs (1d-array): logits
        len_src (int): length of source domain data
        cls_adv (1d-array): adversarial logits of two domains
        srcweight (float): source domain weight for mdd loss

    Returns:
        loss (float): mdd loss
    """
    class_criterion = torch.nn.CrossEntropyLoss()
    y_pred = outputs.max(1)[1]
    target_adv_src = y_pred[:len_src]
    target_adv_tgt = y_pred[len_src:]
    classifier_loss_adv_src = class_criterion(
        cls_adv[:len_src], target_adv_src)
    logloss_tgt = torch.log(torch.clamp(
        1 - F.softmax(cls_adv[len_src:], dim=1), min=1e-15))
    classifier_loss_adv_tgt = F.nll_loss(logloss_tgt, target_adv_tgt)
    loss = srcweight * classifier_loss_adv_src + classifier_loss_adv_tgt
    return loss