"""
cosFormer: Rethinking Softmax In Attention
https://arxiv.org/abs/2202.08791
"""
import math

import numpy as np
import torch
import torch.nn as nn

from layers.ffn import FFN
from utils.config import Config


def add_args(parser):
    parser.add_argument("--cos_act", type=str, default="relu",
                        choices=("relu", "gelu"),
                        help="Activation function for cosformer",
    )

class CosMultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.1, max_len=512):
        super(CosMultiHeadAttention, self).__init__()

        self.n_heads = n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.act_fun = nn.ReLU()
        self.max_len = max_len
        self.weight_index = self._get_index(max_len)

        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, eps=1e-6):
        if k is None:
            k = q
        if v is None:
            v = q

        batch_size, tgt_len, _ = q.size()
        src_len = k.size(1)

        # batch_size x seq_len x n_heads x d
        q = self.q_linear(q).reshape(batch_size, tgt_len, self.n_heads, -1)
        k = self.k_linear(k).reshape(batch_size, src_len, self.n_heads, -1)
        v = self.v_linear(v).reshape(batch_size, src_len, self.n_heads, -1)

        # batch_size x n_heads x seq_len x d
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 1. non-negativity
        q = self.act_fun(q)
        k = self.act_fun(k)

        # 2. cos transform
        assert self.max_len >= max(src_len, tgt_len)
        weight_index = self.weight_index.to(q.device)
        q_index = weight_index[:, :, :tgt_len, :]
        # batch_size x n_heads x seq_len x 2*d_q
        q_ = torch.cat([q * torch.sin(q_index), q * torch.cos(q_index)], dim=-1)
        k_index = weight_index[:, :, :src_len, :]
        # batch_size x n_heads x seq_len x 2*d_k
        k_ = torch.cat([k * torch.sin(k_index), k * torch.cos(k_index)], dim=-1)

        # batch_size x n_heads x 2*d_k * d_v
        kv_ = torch.einsum("bnld,bnlm->bndm", k_, v)
        # Row scaling: batch_size x n_heads x tgt_len
        z_ = 1 / torch.clamp_min(
            torch.einsum('bnld,bnd->bnl', q_,torch.sum(k_, axis=2)), eps)
        # qkl: batch_size x n_heads x tgt_len x d_v
        attn_output = torch.einsum('bnld,bndm,bnl->bnlm', q_, kv_, z_)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, tgt_len, -1)

        # output
        return self.dropout(self.out(attn_output))

        
    def _get_index(self, m):
        # pi * index / 2m
        index = torch.arange(1, m+1, dtype=torch.float32)
        index = np.pi * index / (2 * m)
        index = index.reshape(1, 1, -1, 1)
        return nn.Parameter(index, requires_grad=False)
    

class TransformerLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, max_len=512):
        super(TransformerLayer, self).__init__()

        self.attn = CosMultiHeadAttention(d_model, n_heads,
                                          dropout=dropout,
                                          max_len=max_len)
        self.norm_1 = nn.LayerNorm(d_model)

        self.ffn = FFN(d_model, d_ff, dropout=dropout)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, tgt, mem=None):
        output = self.attn(tgt, mem, mem)
        output = self.norm_1(tgt + output)

        output2 = self.ffn(output)
        output2 = self.norm_2(output2 + output)
        return output2


class Cosformer(nn.Module):

    def __init__(self, args:Config, output_dim) -> None:
        super(Cosformer, self).__init__()

        self.n_heads = args.n_heads
        self.n_layers = args.n_layers
        self.d_model = args.d_model
        self.dropout = args.dropout
        self.d_ff = args.d_ff
        self.mem_len = args.mem_len
        self.max_len = max(args.seq_len, args.mem_len)
        self.vocab_size = args.vocab_size
        self.output_size = output_dim
        
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.embedding_scale = math.sqrt(self.d_model)
        self.layers = nn.ModuleList([
            TransformerLayer(self.d_model, self.n_heads, self.d_ff,
                             self.dropout, self.max_len) 
            for _ in range(self.n_layers)
        ])
        self.output = nn.Linear(self.d_model, self.output_size)

    def forward(self, x):
        x = self.embedding(x) * self.embedding_scale
        for layer in self.layers:
            x = layer(x, x)
        output = self.output(x)
        return output
