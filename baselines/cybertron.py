import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from baselines.transformer import TransformerLayer
from layers.embeddings import PositionalEncoding

class SelectionLayer(nn.Module):

    def __init__(self, d_model, num_kernels, kernel_size):
        super(SelectionLayer, self).__init__()
        self.conv1d = nn.Conv1d(d_model, num_kernels, kernel_size, stride=1)
        self.kernel_size = kernel_size

    def forward(self, x):
        """
        (batch_size, in_channel, seq_len) -> (batch_size, in_channel, num_kernels * kernel_size)
        """
        # Apply the convolution
        conv_out = self.conv1d(x)  # shape: (batch_size, num_kernels, seq_len - kernel_size + 1)

        # Apply softmax to get indices of max probabilities
        probs = F.softmax(conv_out, dim=2)
        values, indices = torch.max(probs, dim=2)  # shape: (batch_size, num_kernels)

        # Prepare indices for gather
        indices_ext = indices.unsqueeze(2) + torch.arange(self.kernel_size).to(x.device)
        indices_ext = indices_ext.reshape(x.size(0), 1, -1).repeat(1, x.size(1), 1)

        # Use gather to extract chunks based on the max indices
        # (batch_size, in_channel, num_kernels * kernel_size)
        output = torch.gather(x, 2, indices_ext)
        log_probs = torch.log(values) # (batch_size, num_kernels)
        return output, log_probs

class AbstractionLayer(nn.Module):

    def __init__(self, d_model, num_kernels, kernel_size, n_heads, n_layers, d_ff, dropout=0.1):
        super(AbstractionLayer, self).__init__()
        self.selection = SelectionLayer(d_model, num_kernels, kernel_size)
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x:torch.Tensor):
        # x of size (batch_size, seq_len, d_model)
        y, log_probs = self.selection(x.transpose(1, 2))
        y = y.transpose(1, 2)
        for i, layer in enumerate(self.layers):
            y, _ = layer(y)
        output = self.output(y)
        return output, log_probs


class Cybertron(nn.Module):

    def __init__(self, vocab_size, d_model, num_kernels, kernel_size, n_heads, n_layers, d_ff, dropout=0.1):
        super(Cybertron, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embedding_scale = math.sqrt(d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.abstract = AbstractionLayer(d_model, num_kernels, kernel_size, n_heads, n_layers, d_ff, dropout)
    
    def single_forward(self, x):
        x = self.embedding(x) * self.embedding_scale
        x += self.pos_encoding(x)
        x, log_probs = self.abstract(x)
        x = x.reshape(x.size(0), -1)
        return x, log_probs

    def forward(self, x, y):
        x, x_log_probs = self.single_forward(x)
        y, y_log_probs = self.single_forward(y)

        log_probs = (x_log_probs + y_log_probs) / 2
        output = (F.cosine_similarity(x, y, dim=1) + 1) / 2
        return output, log_probs


def reinforce_loss(outputs, labels, log_probs):
    # Round the values to the nearest integer
    rounded_values = outputs.round()

    # Compare rounded values with labels and assign 1 if same, -1 otherwise
    rewards = torch.where(rounded_values == labels, 1, -1).unsqueeze(1)
    
    log_probs *= rewards
    # log_probs.sum(dim=1)
    return log_probs.mean()


def train_cybertron(model:Cybertron, train_loader, val_loader, epochs, device, optimizer, init_epoch):

    criterion = nn.BCELoss()

    for epoch in range(init_epoch, epochs):

        train_loss = []

        model.train()
        iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for x, y, label in iterator:

            # zero the parameter gradients
            optimizer.zero_grad()

            x = x.to(device)
            y = y.to(device)
            label = label.to(device)

            outputs, log_probs = model(x, y)
            y = y.reshape(-1)

            loss = criterion(outputs, label) + reinforce_loss(outputs, label, log_probs)
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y, label in val_loader:
                x = x.to(device)
                y = y.to(device)
                label = label.to(device)

                outputs, log_probs = model(x, y)
                
                outputs = outputs.round()
                total += outputs.size(0)
                correct += (outputs == label).sum().item()

        accuracy = 100 * correct / total
        train_losses = np.mean(train_loss)
        print("Epoch {0}: Train Loss: {1:.7f} Accuracy: {2:.2f}% Total{3}" \
                .format(epoch + 1, train_losses, accuracy, total))

    


