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

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

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
    metr = {'f1': f1, 'r': rec, 'p': prec, 'auc': auc, 'acc': acc}
    return metr