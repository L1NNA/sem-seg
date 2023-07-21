"""
Expire Span
http://arxiv.org/abs/2105.06548
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.embeddings import random_position_embedding
from layers.ffn import FFN
from layers.masking import create_masking


def add_args(parser):
    parser.add_argument(
        "--expire_span_alpha",
        type=float,
        default=0,
        help="loss coefficient for reducing spans",
    )
    parser.add_argument(
        "--expire_span_ramp",
        type=int,
        default=32,
        help="ramp length of the masking function",
    )
    parser.add_argument(
        "--expire_span_pre_div",
        type=float,
        default=1.0,
        help="divide activations before non-linearity",
    )
    # parser.add_argument(
    #     "--expire-span-noisy", action="store_true", default=False,
    #     help="whether to randomly forget memories"
    # )

class ExpireSpanDrop(nn.Module):
    """Section 4 of original paper

    Args:
        d_model: hidden dimension
        max_len: T
        ramp: R
        alpha: coefficient for loss
        pre_div: for extreme large spans
    """

    def __init__(self, d_model, mem_len, ramp, alpha, pre_div=1.0):
        super(ExpireSpanDrop, self).__init__()
        self.pre_div = pre_div
        self.mem_len = mem_len
        self.ramp = ramp
        self.alpha = alpha
        self.span_predictor = nn.Linear(d_model, 1)

    def forward(self, attn, mem, counter):
        seq_len = attn.size(1)

        # mem: batch_size x span_len x d_model
        span_predict = self.span_predictor(mem) / self.pre_div
        # e: batch_size x span_len
        e = torch.sigmoid(span_predict) * self.mem_len
        r = e - counter # - (t - i)

        # TODO: add noise to r

        # r_offset: batch_size x seq_len x span_len
        r_offset = r.unsqueeze(1).expand(-1, seq_len, -1)
        offset = torch.arange(seq_len, dtype=r.dtype, device=r.device)
        # align positions for the current sequence
        r_offset = r_offset - offset.reshape(1, -1, 1)

        # mask: batch_size x seq_len x span_len
        m = r_offset / self.ramp + 1.0
        m = torch.clamp(m, 0, 1)

        # calculate loss
        # following paper but different from code
        ramp_mask = (m > 0) * (m < 1)
        span_loss = r_offset * ramp_mask.float()
        loss = span_loss.sum(dim=-1).sum(dim=-1)
        loss = loss / self.mem_len
        loss = loss * self.alpha
        return m, r, loss

class SeqAttention(nn.Module):

    def __init__(self, seq_len, mem_len, d_model, ramp, alpha, pre_div=1.0):
        super(SeqAttention, self).__init__()
        self.expire_span = ExpireSpanDrop(d_model, mem_len, ramp, alpha, pre_div)
        max_len = seq_len + mem_len + ramp
        self.key_pe = random_position_embedding((1, d_model, max_len))
        causal_masking = create_masking(seq_len, max_len)
        self.register_buffer("causal_masking", causal_masking)

    def forward(self, q, k, v, mem, counter):
        _, span_len, d_k = k.size()

        # batch_size x seq_len x span_len
        attn = torch.matmul(q, k.transpose(-1, -2))
        mask, r, loss = self.expire_span(attn, mem, counter)
        # Mask out attention to future steps
        mask = mask.masked_fill(self.causal_masking[:, -span_len:], 0)

        # TODO: add positional embedding
        # attn_pos = torch.matmul(q, self.key_pe)
        # attn = attn + attn_pos[:, :, -span_len:]

        attn = attn.masked_fill(mask == 0, -1e9)
        attn = attn / math.sqrt(d_k)
        attn = F.softmax(attn, dim=-1)
        attn = attn * mask # forget memories
        attn = attn / (attn.sum(-1, keepdim=True) + 1e-8)
        output = torch.matmul(attn, v)
        return output, r, loss

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads, seq_len, mem_len, ramp, alpha, pre_div=1.0, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.d_k = d_model // n_heads
        self.num_heads = n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.attention = SeqAttention(seq_len, mem_len, self.d_k, ramp, alpha, pre_div)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, counter):
        batch_size = q.size(0)

        q = self.q_linear(q).reshape(batch_size, -1, self.num_heads, self.d_k)
        k = self.k_linear(k).reshape(batch_size, -1, self.num_heads, self.d_k)
        v = self.v_linear(v).reshape(batch_size, -1, self.num_heads, self.d_k)

        q = q.transpose(1, 2).reshape(batch_size * self.num_heads, -1, self.d_k)
        k = k.transpose(1, 2).reshape(batch_size * self.num_heads, -1, self.d_k)
        v = v.transpose(1, 2).reshape(batch_size * self.num_heads, -1, self.d_k)

        output, r, loss = self.attention(q, k, v, k, counter)
        output = output.reshape(batch_size, self.num_heads, -1, self.d_k) \
                  .transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.d_k)

        output = self.out(output)
        output = self.dropout(output)
        return output, r, loss
    

class TransformerLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_ff, seq_len, mem_len, ramp, alpha, pre_div=1.0, dropout=0.1):
        super(TransformerLayer, self).__init__()

        self.attn = MultiHeadAttention(d_model, n_heads, seq_len, mem_len, ramp, alpha, pre_div, dropout)
        self.norm_1 = nn.LayerNorm(d_model)

        self.ffn = FFN(d_model, d_ff, dropout=dropout)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, x, mem, counter):
        output, r, loss = self.attn(x, mem, mem, counter)
        output = self.norm_1(x + output)

        output2 = self.ffn(output)
        output2 = self.norm_2(output2 + output)
        return output2, r, loss


class ExpireSpan(nn.Module):

    def __init__(self, args):
        super(ExpireSpan, self).__init__()

        self.n_heads = args.n_heads
        self.n_layers = args.n_layers
        self.d_model = args.d_model
        self.dropout = args.dropout
        self.d_ff = args.d_ff
        self.mem_len = args.mem_len
        self.seq_len = args.seq_len
        self.span_len = args.seq_len + args.mem_len
        self.args = args

        self.embedding = nn.Embedding(args.vocab_size, self.d_model)
        self.layers = nn.ModuleList(
            [TransformerLayer(self.d_model, self.n_heads, self.d_ff, args.seq_len,
                              args.max_len, args.expire_span_ramp,
                              args.expire_span_alpha, args.expire_span_pre_div,
                              self.dropout)
             for _ in range(self.n_layers)]
        )
        self.output = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, x, mems):
        batch_size, seq_len, _ = x.size(1)

        # x: batch_size x seq_len x d_model
        h = self.embedding(x) # no scaling

        h_prev = mems[:-self.n_layers]
        c_prev = mems[-self.n_layers:]
        h_cache = []
        c_cache = []
        aux_loss = 0
        counter = torch.linspace(0, -seq_len+1, steps=seq_len).to(x.device)
        counter = counter.reshape(1, -1).expand(batch_size, -1)  # batch_size x seq_len
        for l, layer in enumerate(self.layers):
            h_cache.append(torch.cat([h_prev[l], h], dim=1))
            c_cache.append(torch.cat([c_prev[l], counter], dim=1))
            h, r, loss = layer(h, h_cache[l], c_cache[l]) 
            aux_loss = aux_loss + loss
            self._drop_memories(h_cache, c_cache, r, seq_len, batch_size, l)
            c_cache[l] += seq_len
        h_cache.extend(c_cache)

        output = self.output(h)

        return output, h_cache, aux_loss
    
    def _drop_memories(self, h_cache, c_cache, r, seq_len, batch_size, l):
        """Determine which memories can be dropped
        """

        # Extend spans by the ramp length R because memories are still
        # used during those R steps.
        r = r + self.args.expire_span_ramp
        # Since spans are measured from the 1st query of this block,
        # subtract seq_len so that they're measured from the next block.
        r = r - seq_len
        r = (r > 0).float()  # batch_size x seq_len
        r[:, -seq_len:].fill_(1) # keep the current memory

        # find the samllest amout to drop
        num_drop = (r <= 0).long().sum(-1)
        num_drop_min = num_drop.min().item()
        # dropping arbitrary numbers might cause memory fragmentation,
        # so only drop with increments of mem_sz. Using mem_sz will
        # ensure that the memory size stay within the limit.
        num_drop_min = int(
            math.floor(num_drop_min / self.seq_len) * self.seq_len
        )
        if num_drop_min != 0:
            spans_sorted, indices = r.sort(dim=-1)
            # from 0 to 1
            spans_sorted[:, num_drop_min:] = 1
            span_mask = torch.zeros_like(r)
            span_mask.scatter_(-1, indices, spans_sorted)
            span_mask = span_mask.bool()
            # batch_size x span_len
            c_cache[l] = c_cache[l][span_mask].reshape(batch_size, -1)
            # batch_size x span_len x d_model
            h_cache[l] = h_cache[l][span_mask].reshape(batch_size, -1, self.d_model)

    def _drop_memories_causal(self, h_cache, c_cache, l, mem_len):
        # batch_size x mem_len x d_model
        h_cache[l] = h_cache[l][:, -mem_len:]
        c_cache[l] = c_cache[l][:, -mem_len:] 
    
    def init_hid_cache(self, batch_size):
        hid = []
        for _ in self.layers:
            h = torch.zeros(batch_size, self.seq_len, self.d_model)
            hid.append(h.to(self.args.device))

        hid.extend(self.init_counter_cache(batch_size))
        return hid

    def init_counter_cache(self, batch_size):
        counter = []
        for _ in self.layers:
            h = torch.linspace(self.seq_len, 1, steps=self.seq_len)
            h = h.reshape(1, -1).expand(batch_size, -1)
            counter.append(h.to(self.args.device))
        return counter