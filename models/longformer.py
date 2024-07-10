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
        self.cls_token_id = get_tokenizer().cls_token_id

        bert_config = LongformerConfig.from_pretrained(LONGFORMER)
        self.encoder = LongformerModel.from_pretrained(LONGFORMER,
                                                    config=bert_config,
                                                    add_pooling_layer=False)
        if self.config.data != 'siamese_clone':
            self.output = nn.Linear(bert_config.hidden_size, self.output_size)

    def _concat_cls_token(self, x:torch.Tensor):
        b = x.size(0)
        x = x.reshape(b, self.n_windows, -1)
        cls_tokens = torch.full((b, self.n_windows, 1), self.cls_token_id).to(x.device)
        x = torch.cat([cls_tokens, x], dim=2)
        x = x.reshape(b, -1)
        win_len = x.size(1) // self.n_windows
        
        global_attention_mask  = torch.zeros(x.shape, dtype=torch.long, device=x.device)
        global_attention_mask [:, [i*win_len for i in range(self.n_windows)]] = 1
        return x, global_attention_mask

    def forward(self, x:torch.Tensor):
        # add cls token at the beginning
        x, global_attention_mask  = self._concat_cls_token(x)

        y = self.encoder(x, global_attention_mask=global_attention_mask).last_hidden_state
        y = cls_pooling(y)
        if self.config.data != 'siamese_clone':
            y = self.output(y)
        return y