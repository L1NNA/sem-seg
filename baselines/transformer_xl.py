"""
Transformer-XL
https://arxiv.org/abs/1901.02860
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers.ffn import FFN
from layers.embeddings import PositionalEmbedding
from layers.masking import create_masking


def add_args(parser):
    pass

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.d_k = d_model // n_heads
        self.num_heads = n_heads

        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.r_linear = nn.Linear(d_model, d_model, bias=False)

        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, r, ru, rv, masking=None, memory=None):
        """
        Args:
            x: current input
            r: relative positional embedding
            ru: relative trainable positional parameters
            rv: relative trainable positional parameters
            masking: attention masking
            memory: previous hidden states
        """
        batch_size = x.size(0)

        q, k, v = x, x, x
        if memory is not None:
            k = torch.cat([memory, k], 1)
            v = torch.cat([memory, v], 1)

        # batch_size x seq_len (+ mem_len) x n_heads x d_k
        qw = self.q_linear(q).reshape(batch_size, -1, self.num_heads, self.d_k)
        kw = self.k_linear(k).reshape(batch_size, -1, self.num_heads, self.d_k)
        vw = self.v_linear(v).reshape(batch_size, -1, self.num_heads, self.d_k)
        # seq_len (+ mem_len) x n_heads x d_k
        rw = self.r_linear(r).reshape(-1, self.num_heads, self.d_k)

        # See section 3.3 in Transformer-XL paper for r, ru, rv
        # batch_size x n_heads x seq_len x seq_len (+ mem_len)
        AC = torch.einsum('bind,bjnd->bnij', (qw + ru, kw))
        # batch_size x n_heads x seq_len x seq_len (+ mem_len)
        BD = torch.einsum('bind,jnd->bnij', (qw + rv, rw))
        BD = self._shift(BD)
        attn_score = AC + BD
        if masking is not None:
            attn_score = attn_score.masked_fill(masking == 0, -1e9)

        # attention
        attn_score = attn_score / math.sqrt(self.d_k)
        attn_prob = F.softmax(attn_score, dim=-1)
        output = torch.einsum('bnij,bjnd->bind', (attn_prob, vw))
        output = output.reshape(batch_size, -1, self.num_heads * self.d_k)

        output = self.out(output)
        output = self.dropout(output)

        return output
    
    def _shift(self, x):
        """
        Shift the ith row by n - i - 1 to the left,
        where n is the length of the row.
        (See Appendix B in Transformer-XL paper)
        """
        # (batch_size, n_heads, seq_len, seq_len + mem_len + 1)
        x_padded = F.pad(x, (1, 0), value=0)
        # (batch_size, n_heads, seq_len + mem_len + 1, seq_len)
        x_padded = x_padded.reshape(x.size(0), x.size(1), x.size(3) + 1, x.size(2))
        # -> (batch_size, n_heads, seq_len, seq_len + mem_len)
        return x_padded[:, :, 1:].reshape_as(x)


class TransformerXLLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerXLLayer, self).__init__()

        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.norm_1 = nn.LayerNorm(d_model)

        self.ffn = FFN(d_model, d_ff, dropout=dropout)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, x, pos, ru, rv, masking=None, mems=None):
        output = self.attn(x, pos, ru, rv, masking, mems)
        output = self.norm_1(x + output)

        output2 = self.ffn(output)
        output2 = self.norm_2(output2 + output)
        return output2


class TransformerXL(nn.Module):

    def __init__(self, args):
        super(TransformerXL, self).__init__()

        self.n_heads = args.n_heads
        self.n_layers = args.n_layers
        self.d_model = args.d_model
        self.dropout = args.dropout
        self.d_ff = args.d_ff
        self.mem_len = args.mem_len
        self.seq_len = args.seq_len

        self.embedding = nn.Embedding(args.vocab_size, self.d_model)
        self.pos_emb = PositionalEmbedding(self.d_model)

        self.layers = nn.ModuleList(
            [TransformerXLLayer(self.d_model, self.n_heads,
                                self.d_ff, self.dropout)
             for _ in range(self.n_layers)]
        )

        self._create_params()

    def forward(self, x, memory=None):
        if not memory:
            memory = self._init_memory(x)

        span_len = x.size(1)
        if memory is not None and len(memory[0].size()) > 1:
            span_len += memory[0].size(1)

        x = self.embedding(x)
        mask = create_masking(self.seq_len, span_len, x.device)
        pos_emb = self.pos_emb.relative(span_len, x.device)

        hids = []
        hids.append(x)
        for l, layer in enumerate(self.layers):
            # the l-1th memory from previous sequence,
            # see section 3.2 in Transformer-XL paper
            memory_l = None if memory is None else memory[l]
            x = layer(x, pos_emb, self.ru, self.rv, mask, memory_l)
            hids.append(x)
        new_memory = self._update_memory(hids, memory, self.mem_len)
        return x, new_memory
    
    def _update_memory(self, hids, memory, mem_len):
        # does not deal with None
        if memory is None:
            return None

        # mems is not None
        assert len(hids) == len(memory), 'len(hids) != len(mems)'

        with torch.no_grad():
            new_mems = []
            for i in range(len(hids)):
                cat = torch.cat([memory[i], hids[i]], dim=1)
                new_mems.append(cat[:, -mem_len:].detach())

        return new_mems
    
    def _init_memory(self, x):
        if self.mem_len > 0:
            mems = []
            for _ in range(self.n_layers + 1):
                empty = torch.empty(0, dtype=x.dtype, device=x.device)
                mems.append(empty)
            return mems
        else:
            return None
    
    def _create_params(self):
        self.dk = self.d_model // self.n_heads
        self.ru = nn.Parameter(torch.Tensor(self.n_heads, self.dk))
        self.rv = nn.Parameter(torch.Tensor(self.n_heads, self.dk))