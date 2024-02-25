import torch
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix


def confusion(labels, predictions, num_of_classes):
    total = labels.size(0)
    cm = confusion_matrix(labels, predictions)

    accuracy = 100 * np.sum(np.diag(cm)) / total

    print(cm)

    true_positives = cm[1, 1]
    false_positives = cm[0, 1]
    false_negatives = cm[1, 0]
    true_negatives = cm[0, 0]
    
    precision = true_positives / (true_positives + false_positives + 1e-5)
    recall = true_positives / (true_positives + false_negatives + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    return accuracy, precision, recall, f1


def calculate_auroc(labels, logits):
    labels = labels.tolist()
    num_of_classes = 1 if len(logits.size()) == 1 else logits.size(1)
    if num_of_classes == 2:
        logits = logits[:, 1]
    logits = logits.tolist()
    if num_of_classes > 2:
        return roc_auc_score(labels, logits, multi_class='ovr')
    else:
        return roc_auc_score(labels, logits)


def number_of_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
