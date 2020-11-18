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
            precision, recall, thresholds = precision_recall_curve(y_true, y_score)
            np.savetxt('prec1.csv', precision, fmt='%.2f', delimiter=',')
            np.savetxt('recall1.csv', recall, fmt='%.2f', delimiter=',')
            plt.plot(precision, recall, marker='o')
            plt.savefig('pr_curve1.png')
            
            plt.clf()
            fpr, tpr, threshold = roc_curve(y_true, y_score, pos_label=1)
            np.savetxt('fpr1.csv', fpr, fmt='%.2f', delimiter=',')
            np.savetxt('tpr1.csv', tpr, fmt='%.2f', delimiter=',')
            plt.plot(fpr, tpr, marker='o')
            plt.savefig('roc_curve1.png')
    acc = accuracy_score(y_true, y_pred)
    metr = {'f1': f1, 'r': rec, 'p': prec, 'auc': auc, 'accuracy': acc}
    return metr
