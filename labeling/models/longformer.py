"""
Longformer: The Long-Document Transformer
https://arxiv.org/abs/2004.05150
"""
import torch
import torch.nn as nn
from transformers import LongformerConfig, LongformerModel

from utils.setup_BPE import LONGFORMER, get_tokenizer
from layers.pooling import cls_pooling
from utils.config import Config


def add_args(_):
    pass

class LongFormer(nn.Module):

    def __init__(self, config:Config, output_dim) -> None:
        super(LongFormer, self).__init__()
        self.config = config
        self.output_size = output_dim

        bert_config = LongformerConfig.from_pretrained(LONGFORMER)
        self.encoder = LongformerModel.from_pretrained(LONGFORMER,
                                                    config=bert_config,
                                                    add_pooling_layer=False)

        self.output = nn.Linear(bert_config.hidden_size, self.output_size)

    def forward(self, x:torch.Tensor):
        # start from 1 since 0 is for bos token
        y = self.encoder(x).last_hidden_state
        y = self.output(y)
        y = cls_pooling(y)
        return y