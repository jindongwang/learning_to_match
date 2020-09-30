import torch
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

def metric(y_true, y_pred, y_score):
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_score)
    metr = {'f1': f1, 'r': rec, 'p': prec, 'auc': auc}
    return metr