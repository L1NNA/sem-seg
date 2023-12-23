import torch
from sklearn.metrics import roc_auc_score


def confusion(labels, predictions):
    total = labels.size(0)

    true_lables = labels.sum().item()
    false_lables = total - true_lables
    true_positives = (predictions * labels).sum().item()
    false_positives = (predictions * (1 - labels)).sum().item()
    false_negatives = ((1 - predictions) * labels).sum().item()
    true_negatives = ((1 - predictions) * (1 - labels)).sum().item()

    accuracy = 100 * (true_positives + true_negatives) / total
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, f1


def calculate_auroc(labels, logits):
    labels = labels.tolist()
    logits = logits.tolist()

    return roc_auc_score(labels, logits)


def number_of_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
