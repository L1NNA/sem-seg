import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=15000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2, dtype=torch.float32) * - (math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].to(x.device)
    
    def relative(self, span_len, device):
        return torch.flip(self.pe[:, :span_len], [1]).to(device)
    

def random_position_embedding(sizes):
    return nn.Parameter(torch.randn(*sizes))