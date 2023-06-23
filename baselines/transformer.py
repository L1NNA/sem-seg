import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers.ffn import FFN
from layers.embeddings import PositionalEmbedding
from layers.masking import create_masking


def add_args(parser):
    parser.add_argument(
        "--n_heads", type=int, default=4, help="number of attention heads"
    )
    parser.add_argument(
        "--n_layers", type=int, default=6, help="number of layers"
    )
    parser.add_argument(
        "--d_model", type=int, default=512, help="hidden dimension"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="dropout rate"
    )
    parser.add_argument(
        "--d_ff", type=int, default=1024, help="feed forward hidden dimension"
    )


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
        output = output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.d_k)

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


class Transformer(nn.Module):

    def __init__(self, args) -> None:
        super(Transformer, self).__init__()

        self.n_heads = args.n_heads
        self.n_layers = args.n_layers
        self.d_model = args.d_model
        self.dropout = args.dropout
        self.d_ff = args.d_ff
        
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_embedding = PositionalEmbedding(self.d_model)
        self.layers = nn.ModuleList([
            TransformerLayer(self.d_model, self.n_heads, self.d_ff, self.dropout) for _ in range(self.n_layers)
        ])
        self.output = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, x, masking=None):
        x = self.embedding(x)
        x += self.pos_embedding(x)
        if masking is None:
            seq_len = x.size(1)
            masking = create_masking(seq_len, seq_len, x.device)
        attns = {}
        for i, layer in enumerate(self.layers):
            x, attn = layer(x, masking=masking)
            attns[f'attn_{i}'] = attn
        output = self.output(x)
        return output, attns