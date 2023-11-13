import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers.ffn import FFN
from layers.embeddings import PositionalEncoding


def scaled_dot_product_attention(q, k, v, masking=None):
    d_k = k.size(-1)
    weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if masking is not None:
        weights = weights.masked_fill(masking, -1e9)
    attn = F.softmax(weights, dim=-1)
    output = torch.matmul(attn, v)
    return output, attn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.d_k = d_model // n_heads
        self.num_heads = n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, masking=None):
        batch_size = q.size(0)

        q = self.q_linear(q).reshape(batch_size, -1, self.num_heads, self.d_k)
        k = self.k_linear(k).reshape(batch_size, -1, self.num_heads, self.d_k)
        v = self.v_linear(v).reshape(batch_size, -1, self.num_heads, self.d_k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        output, attn = scaled_dot_product_attention(q, k, v, masking=masking)
        output = output.transpose(1, 2) \
                    .reshape(batch_size, -1, self.num_heads * self.d_k)

        output = self.out(output)
        output = self.dropout(output)
        return output, attn
    

class TransformerLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerLayer, self).__init__()

        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.norm_1 = nn.LayerNorm(d_model)

        self.ffn = FFN(d_model, d_ff, dropout=dropout)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, x, masking=None):
        output, attn = self.attn(x, x, x, masking=masking)
        output = self.norm_1(x + output)

        output2 = self.ffn(output)
        output2 = self.norm_2(output2 + output)
        return output2, attn