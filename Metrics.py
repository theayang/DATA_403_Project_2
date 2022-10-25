from sklearn.metrics import roc_auc_score
import numpy as np

def get_accuracy(p, Y):
    return sum((p > .5) == Y) / len(Y)

def get_precision(p, Y):
    # precision = TP / (TP + FP)
    TP = np.where(Y, ((p > .5) == Y), False).sum()
    FP = np.where(Y == 0, (p > .5), False).sum()
    return TP / (TP + FP)
   
def get_recall(p, Y):
    # recall = TP / (TP + FN)
    TP = np.where(Y, ((p > .5) == Y), False).sum()
    FN = np.where(Y, ((p > .5) != Y), False).sum()
    return TP / (TP + FN)

def get_f1(p, Y):
    precision = get_precision(p, Y)
    recall = get_recall(p, Y)
    return 2 / ((1 / precision) + (1 / recall))
    
# Compute Accuracy, precision, recall, and f1
def compute_metrics(p, Y):
    # precision = TP / (TP + FP)
    # recall = TP / (TP + FN)
    accuracy = sum((p > .5) == Y) / len(Y)
    
    TP = np.where(Y, ((p > .5) == Y), False).sum()
    FP = np.where(Y == 0, (p > .5), False).sum()
    precision = TP / (TP + FP)
    
    FN = np.where(Y, ((p > .5) != Y), False).sum()
    recall = TP / (TP + FN)
    
    f1 = 2 / ((1 / precision) + (1 / recall))
    
    auc = roc_auc_score(Y, p)
    print(f"Accuracy: {round(accuracy, 3)}; Precision: {round(precision, 3)}; " + \
           f"Recall: {round(recall, 3)}; f1: {round(f1, 3)}; ROC-AUC: {round(auc, 3)}")