from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np

def metric(y_true, y_pred, y_score):
    prec, rec, f1, auc = 0, 0, 0, 0
    if len(np.unique(y_true)) == 2:
        prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary')
        auc = roc_auc_score(y_true, y_score)
    acc = accuracy_score(y_true, y_pred)
    metr = {'f1': f1, 'r': rec, 'p': prec, 'auc': auc, 'accuracy': acc}
    # cls_report = classification_report(y_true, y_pred)
    # print(cls_report)
    # print(confusion_matrix(y_true, y_pred))
    return metr
