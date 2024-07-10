"""
Two-Level Transformer and Auxiliary Coherence
Modeling for Improved Text Segmentation
https://arxiv.org/abs/2001.00891
"""
import math

import torch
import torch.nn as nn

from layers.transformer import TransformerLayer
from layers.embeddings import PositionalEncoding
from layers.pooling import any_max_pooling, cls_pooling
from utils.config import Config
from utils.setup_BPE import get_tokenizer


def add_args(args):
    args.add_argument("--n_windows", type=int, default=0,
                      help="number of windows (must divide seq_len)")
    args.add_argument("--w_layers", type=int, default=6,
                      help="number of layers for window encoder")
    # args.add_argument("--coherence_hinge_margin", type=float, default=0)


class CATS(nn.Module):
    
    def __init__(self, config:Config, output_dim):
        super(CATS, self).__init__()
        self.config = config
        self.output_size = output_dim
        self.w = config.n_windows
        assert config.seq_len % self.w == 0, \
            "seq_len must be divisible by n_windows"
        self.s_ = config.seq_len // self.w + 1
        self.w_ = self.w + 1

        # embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model) 
        self.embedding_scale = math.sqrt(config.d_model)
        self.pe = PositionalEncoding(config.d_model)

        # add bos token at the beginning of each window
        self.cls_token = get_tokenizer().cls_token_id

        # Encode each sentence
        self.sent_encoder = nn.ModuleList([
            TransformerLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        # Encode each window
        self.window_encoder = nn.ModuleList([
            TransformerLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.w_layers)
        ])

        # Segmentation Classifier
        if self.output_size > 0:
            self.seg = nn.Linear(config.d_model, self.output_size)

        # Auxiliary Regressor
        # self.aux = nn.Linear(config.d_model, 1)
        # self.aux_softmax = nn.Softmax(dim=2)
        # self.coherence_hinge_margin = config.coherence_hinge_margin

    def _concat_cls_token(self, x):
        cls_tokens = torch.full((x.size(0), 1), self.cls_token).to(x.device) # b x 1
        cls_tensor = self.embedding(cls_tokens) * self.embedding_scale
        cls_tensor = cls_tensor.reshape(x.size(0), 1, x.size(2)) # b x 1 x d
        x = torch.cat([cls_tensor, x], dim=1) # b x s+1 x d
        return x
    
    def forward(self, x:torch.Tensor):
        b = x.size(0)
        w, s_ = self.w, self.s_

        # wording embedding and positional encoding
        x = self.embedding(x) * self.embedding_scale # b x s x d
        x += self.pe(x) # b x s x d

        # add cls token
        x = x.reshape(b*w, s_-1, -1) # (b*w) x (s'-1) x d
        x = self._concat_cls_token(x) # (b*w) x s' x d
        
        # encode each sentence
        for layer in self.sent_encoder:
            # (b*w) x s' x d
            x, _ = layer(x)
        y = cls_pooling(x) # (b*w) x d
        y = y.reshape(b, w, -1) # b x w x d

        # add positional encoding again to windows
        y += self.pe(y) # b x w x d
        y = self._concat_cls_token(y) # b x w+1 x d
        # encode each window (paragraph)
        for layer in self.window_encoder:
            # b x w x d
            y, _ = layer(y)
        # b x w x 2
        if self.output_size > 0:
            y = self.seg(y)

        # calculate auxiliary loss
        # z = self.aux(y)
        # z = self.aux_softmax(z)
        # norm_scores_true = z[:, 0]
        # norm_scores_false = z[:, 1]
        # norm_scores = norm_scores_true - norm_scores_false
        # norm_scores = self.coherence_hinge_margin - norm_scores
        # aux_loss = torch.clamp(norm_scores, min=0)
        z = cls_pooling(y)
        
        return z

