from sklearn.metrics import roc_auc_score
import numpy as np

def accuracy(p, Y):
    return sum((p > .5) == Y) / len(Y)

def precision(p, Y):
    # precision = TP / (TP + FP)
    TP = np.where(Y, ((p > .5) == Y), False).sum()
    FP = np.where(Y == 0, (p > .5), False).sum()
    return TP / (TP + FP)
   
def recall(p, Y):
    # recall = TP / (TP + FN)
    TP = np.where(Y, ((p > .5) == Y), False).sum()
    FN = np.where(Y, ((p > .5) != Y), False).sum()
    return TP / (TP + FN)

def fmeasure(p, Y, B=1):
    precision = precision(p, Y)
    recall = recall(p, Y)
    return ((1 + B ** 2) * precision * recall) / ((B ** 2 * precision) + recall)

def sensitivity(p, Y):
    TP = np.where(Y, ((p > .5) == Y), False).sum()
    FN = np.where(Y, ((p > .5) != Y), False).sum()
    return TP / (TP + FN)

def specificity(p, Y):
    TN = np.where(Y, ((p <= .5) == Y), False).sum()
    FP = np.where(Y == 0, (p > .5), False).sum()
    return TN / (TN + FP)

def roc_auc(p, Y):
    return roc_auc_score(Y, p)
    
# Compute Accuracy, precision, recall, and f1
def compute_metrics(p, Y):
    accuracy = accuracy(p, Y)
    precision = precision(p, Y)
    recall = recall(p, Y)
    f1 = fmeasure(p, Y, B=1)
    auc = roc_auc_score(Y, p)
    print(f"Accuracy: {round(accuracy, 3)}; Precision: {round(precision, 3)}; " + \
           f"Recall: {round(recall, 3)}; f1: {round(f1, 3)}; ROC-AUC: {round(auc, 3)}")

def compute_metrics2(p, Y):
    accuracy = accuracy(p, Y)
    sensitivity = sensitivity(p, Y)
    specificity = specificity(p, Y)
    f1 = fmeasure(p, Y, B=1)
    auc = roc_auc_score(Y, p)
    print(f"Accuracy: {round(accuracy, 3)}; Sensitivity: {round(sensitivity, 3)}; " + \
           f"Specificity: {round(specificity, 3)}; f1: {round(f1, 3)}; ROC-AUC: {round(auc, 3)}")