import torch
import torch.nn as nn
import torch.nn.functional as F


class SegInfoNCELoss(nn.Module):

    def __init__(self, n_windows, temperature=0.05):
        super(SegInfoNCELoss, self).__init__()
        self.temperature = temperature
        self.n_windows = n_windows
        self.cross_entropy = nn.CrossEntropyLoss()

    # @torch.compile
    def forward(self, x_embeddings, y_embeddings):
        batch_size = x_embeddings.size(0)
        if self.n_windows > 1:
            batch_size = batch_size // n_windows

            labels = torch.arange(n_windows, requires_grad=False) \
                .repeat(batch_size).to(x_embeddings.device)
            # Compute cosine similarities: batch_size x n_windows x n_windows
            similarities = F.cosine_similarity(x_embeddings.unsqueeze(2), \
                y_embeddings.unsqueeze(1), dim=3) / self.temperature
            # Convert to logits
            logits = similarities - torch.logsumexp(similarities, dim=2, keepdim=True)
            logits = logits.reshape(batch_size*n_windows, -1)
        else:
            labels = torch.arange(batch_size, requires_grad=False).to(x_embeddings.device)
            # Compute cosine similarities: batch_size x n_windows x n_windows
            similarities = F.cosine_similarity(x_embeddings.unsqueeze(2), \
                y_embeddings.unsqueeze(1), dim=2) / self.temperature
            logits = similarities - torch.logsumexp(similarities, dim=1, keepdim=True)

        # Calculate loss using cross-entropy
        loss = self.cross_entropy(logits, labels)
        return loss
