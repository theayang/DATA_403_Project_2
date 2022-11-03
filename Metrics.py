from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd


def accuracy(p, Y):
    return sum((p > .5) == Y) / len(Y)


def precision(p, Y):
    # precision = TP / (TP + FP)
    TP = np.where(Y, ((p > .5) == Y), False).sum()
    FP = np.where(Y == 0, (p > .5), False).sum()
    if TP + FP == 0:
        return 0
    return TP / (TP + FP)


def recall(p, Y):
    # recall = TP / (TP + FN)
    TP = np.where(Y, ((p > .5) == Y), False).sum()
    FN = np.where(Y, ((p > .5) != Y), False).sum()
    if TP + FN == 0:
        return 0
    return TP / (TP + FN)


def fmeasure(p, Y, B=1):
    prec = precision(p, Y)
    reca = recall(p, Y)
    if prec == 0 and reca == 0:
        return 0
    return ((1 + B ** 2) * prec * reca) / ((B ** 2 * prec) + reca)


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


def net_revenue(p, Y, ann):
    if len(p) != len(ann) or len(Y) != len(ann):
        print('Wrong lengths of inputs')
        return None
    ann = 0
    for i in range(len(Y)):
        if (p[i] > .5) == Y[i] and p[i] > .5:
            ann += 0
        elif (p[i] < .5) == Y[i] and p[i] < .5:
            ann += ann[i]
        else:
            ann -= ann[i]
    return ann

# Compute Accuracy, precision, recall, and f1


def compute_metrics(p, Y):
    accuracy = accuracy(p, Y)
    precision = precision(p, Y)
    recall = recall(p, Y)
    f1 = fmeasure(p, Y, B=1)
    auc = roc_auc_score(Y, p)
    print(f"Accuracy: {round(accuracy, 3)}; Precision: {round(precision, 3)}; " +
          f"Recall: {round(recall, 3)}; f1: {round(f1, 3)}; ROC-AUC: {round(auc, 3)}")


def compute_metrics2(p, Y):
    accuracy = accuracy(p, Y)
    sensitivity = sensitivity(p, Y)
    specificity = specificity(p, Y)
    f1 = fmeasure(p, Y, B=1)
    auc = roc_auc_score(Y, p)
    print(f"Accuracy: {round(accuracy, 3)}; Sensitivity: {round(sensitivity, 3)}; " +
          f"Specificity: {round(specificity, 3)}; f1: {round(f1, 3)}; ROC-AUC: {round(auc, 3)}")


def confusion_matrix(predictions, truth):
    conf = np.array([[0, 0], [0, 0]])
    conf[0, 0] = ((predictions == 1) & (truth == 1)).sum()
    conf[0, 1] = ((predictions == 1) & (truth != 1)).sum()
    conf[1, 0] = ((predictions == 0) & (truth == 1)).sum()
    conf[1, 1] = ((predictions == 0) & (truth == 0)).sum()
    return pd.DataFrame(conf, columns=['True 1', 'True 0'], index=['Pred 1', 'Pred 0'])
