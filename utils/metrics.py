import torch
from sklearn.metrics import roc_auc_score, precision_score, recall_score


def confusion(labels, predictions):
    total = labels.size(0)

    true_lables = labels.sum().item()
    false_lables = total - true_lables
    true_positives = (predictions * labels).sum().item()
    false_positives = (predictions * (1 - labels)).sum().item()
    false_negatives = ((1 - predictions) * labels).sum().item()
    true_negatives = ((1 - predictions) * (1 - labels)).sum().item()

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = 100 * (true_positives + true_negatives) / total
    return accuracy, recall, precision


def calculate_auroc(labels, logits):
    labels = labels.tolist()
    logits = logits.tolist()

    return roc_auc_score(labels, logits)
    
