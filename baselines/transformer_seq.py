"""
Adaptive Span Model (https://arxiv.org/abs/1905.07799)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.ffn import FFN


def add_args(parser):
    parser.add_argument(
        "--ramp_size", type=int, default=4, help="number of attention heads"
    )


class AdaptiveMask(nn.Module):
    """
    m(x) = min(max(0, (R + z - x) / R), 1)
    ramp_size: R
    init_val: z
    """

    def __init__(self, max_size, ramp_size, init_val=0, shape=(1,)):
        super(AdaptiveMask, self).__init__()
        self.max_size = max_size
        self.ramp_size = ramp_size
        self.current_val = nn.Parameter(torch.zeros(*shape) + init_val)
        mask_template = torch.linspace(1 - max_size, 0, steps=max_size)
        self.register_buffer('mask_template', mask_template)

    def forward(self, x):
        # z - x
        mask = self.current_val * self.max_size + self.mask_template
        mask = mask / self.ramp_size + 1
        mask = mask.clamp(0, 1)
        if x.size(-1) < self.max_size:
            # the input could have been trimmed beforehand to save computation
            mask = mask[:, :, -x.size(-1):]
        x = x * mask
        return x

    def get_current_max_size(self, include_ramp=True):
        current_size = math.ceil(self.current_val.max().item() * self.max_size)
        if include_ramp:
            current_size += self.ramp_size
        current_size = max(0, min(self.max_size, current_size))
        return current_size

    def get_current_avg_size(self, include_ramp=True):
        current_size = math.ceil(self.current_val.mean().item() * self.max_size)
        if include_ramp:
            current_size += self.ramp_size
        current_size = max(0, min(self.max_size, current_size))
        return current_size

    def clamp_param(self):
        """this need to be called after each update"""
        self.current_val.data.clamp_(0, 1)


class AdaptiveSpan(nn.Module):

    def __init__(self, span_len, n_heads):
        super(AdaptiveSpan, self).__init__()

    def forward(self, q, k, v, masking=None):
        pass

    def trim_memory(self, q, k, v, k_pe):
        pass

class SeqAttention(nn.Module):

    def __init__(self, span_len, n_heads):
        super(SeqAttention, self).__init__()
        self.adaptive_span = AdaptiveSpan(span_len, n_heads)

    def forward(self, q, k, v, k_pe=None):
        d_k = k.size(-1)

        k, v = self.adaptive_span.trim_memory(q, k, v, k_pe)

        # batch_size x seq_len x seq_len + mem_len
        attn_scores = torch.matmul(q, k.transpose(-1, -2))
        # batch_size x seq_len x seq_len + mem_len
        attn_pos = torch.matmul(q, k_pe) 
        # Relative positional embedding 
        attn_scores = attn_scores + attn_pos
        attn_scores = attn_scores / math.sqrt(d_k)

        attn = F.softmax(attn_scores, dim=-1)
        attn = self.adaptive_span(attn)

        output = torch.matmul(attn, v)

        return output

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads, span_len, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.d_k = d_model // n_heads
        self.num_heads = n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.attention = SeqAttention(span_len, n_heads)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, k_pe):
        batch_size = q.size(0)

        q = self.q_linear(q).reshape(batch_size, -1, self.num_heads, self.d_k)
        k = self.k_linear(k).reshape(batch_size, -1, self.num_heads, self.d_k)
        v = self.v_linear(v).reshape(batch_size, -1, self.num_heads, self.d_k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        output = self.attention(q, k, v, k_pe)
        output = output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.d_k)

        output = self.out(output)
        output = self.dropout(output)
        return output
    

class AdaptiveSpanLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(AdaptiveSpanLayer, self).__init__()

        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.norm_1 = nn.LayerNorm(d_model)

        self.ffn = FFN(d_model, d_ff, dropout=dropout)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, x, mem, k_pe):
        # batch_size x seq_len + mem_len x d_model
        h = torch.cat([mem, h], dim=1)
        output, attn = self.attn(x, h, h, k_pe)
        output = self.norm_1(x + output)

        output2 = self.ffn(output)
        output2 = self.norm_2(output2 + output)
        return output2, attn


class TransformerSeq(nn.Module):

    def __init__(self, args):
        super(TransformerSeq, self).__init__()

        self.n_heads = args.n_heads
        self.n_layers = args.n_layers
        self.d_model = args.d_model
        self.dropout = args.dropout
        self.d_ff = args.d_ff
        self.mem_len = args.mem_len
        self.seq_len = args.seq_len
        self.span_len = args.seq_len + args.mem_len

        self.embedding = nn.Embedding(args.vocab_size, self.d_model)
        # positional embedding
        self.key_pe = nn.Parameter(
            torch.randn(1, 1, self.span_len, self.d_model // self.n_heads))
        self.dropout = nn.Dropout(self.dropout)

        self.layers = nn.ModuleList(
            [AdaptiveSpanLayer(self.d_model, self.n_heads,
                                self.d_ff, self.dropout)
             for _ in range(self.n_layers)]
        )

    def forward(self, x, mems=None):
        if not mems:
            mems = self._init_mems()

        x = self.embedding(x)

        new_mems = []
        new_mems.append(x)
        for i,layer in enumerate(self.layers):
            mems_i = None if mems is None else mems[i]
            x = layer(x, mems_i, self.key_pe)
            new_mems.append(x)
        # new_mems = self._update_mems(new_mems, mems, self.mem_len)
        return x, new_mems
    
    def _init_memory(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for _ in range(self.n_layers + 1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None