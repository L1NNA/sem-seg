"""
CodeT5+: Open Code Large Language Models for Code Understanding and Generation
https://arxiv.org/pdf/2305.07922
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from layers.transformer import FFN
from utils.setup_BPE import CODET5P, get_tokenizer
from utils.config import Config


def add_args(_):
    pass

class CodeT5P(nn.Module):
    """
    the hyperparameters are fixed as the followings:
    vocab_size=32103
    d_model=768
    num_layers=
    num_heads=
    d_ff=
    seq_len=
    """

    def __init__(self, config:Config, output_dim) -> None:
        super(CodeT5P, self).__init__()
        self.config = config
        self.output_size = output_dim
        self.emb_dim = 256
        self.encoder = AutoModel.from_pretrained(CODET5P, trust_remote_code=True)
        if self.output_size > 0:
            self.output = nn.Linear(self.emb_dim, self.output_size)

    def forward(self, x:torch.Tensor, masking:torch.Tensor):
        y = self.encoder(x, attention_mask=masking)
        if self.output_size > 0:
            y = self.output(y)
        return y