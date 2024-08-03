import torch
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix


def confusion(labels, predictions):
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


def calculate_mrr(tensor, masking=None):
    """
    Calculate the Mean Reciprocal Rank (MRR) for a batch of data where each row contains relevance scores.
    
    Args:
    tensor (torch.Tensor): A tensor of shape (batch, batch) where the relevant item for each query is at the diagonal position (i, i).
    
    Returns:
    float: The Mean Reciprocal Rank (MRR).
    """
    batch_size = tensor.shape[0]
    
    # Initialize a list to store ranks
    ranks = []

    for i in range(batch_size):
        if masking is not None and masking[i] == 0:
            continue
        # Get the relevance scores for the i-th row
        relevance_scores = tensor[i]
        
        # Get the score of the relevant item (diagonal element)
        relevant_score = relevance_scores[i]
        
        # Calculate the rank of the relevant score within the row
        rank = (relevance_scores > relevant_score).sum().item() + 1
        
        # Append the rank to the list
        ranks.append(rank)
    
    # Convert ranks to a tensor
    ranks = torch.tensor(ranks).float()
    
    # Calculate the reciprocal ranks
    reciprocal_ranks = 1 / ranks
    
    # Compute the Mean Reciprocal Rank (MRR)
    mrr = reciprocal_ranks.mean().item()
    
    return mrr


    
