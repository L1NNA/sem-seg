"""
GraphCodeBERT: Pre-training Code Representations with Data Flow
https://arxiv.org/abs/2009.08366
"""
import torch
import torch.nn as nn
from transformers import LongformerConfig, LongformerModel

from utils.setup_BPE import LONGFORMER, get_tokenizer
from layers.pooling import cls_pooling
from utils.config import Config


def add_args(_):
    pass

class Longformer(nn.Module):
    """
    the hyperparameters are fixed as the followings:
    vocab_size=50265
    d_model=768
    num_layers=12
    num_heads=12
    d_ff=3072
    seq_len=4096
    """

    def __init__(self, config:Config, output_dim) -> None:
        super(Longformer, self).__init__()
        self.config = config
        self.output_size = output_dim
        self.n_windows = config.n_windows

        bert_config = LongformerConfig.from_pretrained(LONGFORMER)
        self.encoder = LongformerModel.from_pretrained(LONGFORMER,
                                                    config=bert_config,
                                                    add_pooling_layer=False)
        if self.output_size > 0:
            self.output = nn.Linear(bert_config.hidden_size, self.output_size)

    def _global_attn_ids(self, x:torch.Tensor, x_mask:torch.Tensor):
        win_len = x.size(1) // self.n_windows
        global_attention_mask  = torch.zeros(x.shape, dtype=torch.long, device=x.device)
        global_attention_mask[:, [i*win_len for i in range(self.n_windows)]] = 1
        x_mask[:, [i*win_len for i in range(self.n_windows)]] = 1
        return x, global_attention_mask, x_mask

    def forward(self, x:torch.Tensor, x_mask:torch.Tensor):
        b = x.size(0)
        win_len = x.size(1) // self.n_windows
        # add cls token at the beginning
        x, global_attention_mask, x_mask = self._global_attn_ids(x, x_mask)

        y = self.encoder(x, global_attention_mask=global_attention_mask, attention_mask=x_mask).last_hidden_state
        y = y.reshape(b*self.n_windows, win_len, -1)
        y = cls_pooling(y)
        if self.output_size > 0:
            y = self.output(y)
        return y