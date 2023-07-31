"""
Two-Level Transformer and Auxiliary Coherence
Modeling for Improved Text Segmentation
https://arxiv.org/abs/2001.00891
"""
import math

import torch
import torch.nn as nn

from baselines.transformer import TransformerLayer
from layers.embeddings import PositionalEncoding
from layers.masking import create_masking
from layers.pooling import any_max_pooling
from utils.config import Config
from data_loader.setup_BPE import get_tokenizer


def add_args(args):
    args.add_argument("--n_windows", type=int, default=8,
                      help="number of windows (must divide seq_len)")
    args.add_argument("--w_layers", type=int, default=6,
                      help="number of layers for window encoder")
    # args.add_argument("--coherence_hinge_margin", type=float, default=0)


class CATS(nn.Module):
    
    def __init__(self, config:Config, output_dim):
        super(CATS, self).__init__()
        self.config = config
        assert output_dim == 2, "CATS only supports binary classification"
        self.output_size = output_dim
        self.w = config.n_windows
        assert config.seq_len % self.w == 0, \
            "seq_len must be divisible by n_windows"
        self.s_ = config.seq_len // self.w + 1

        # embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model) 
        self.embedding_scale = math.sqrt(config.d_model)
        self.pe = PositionalEncoding(config.d_model)
        self.cls_tokens = torch.tensor([get_tokenizer().bos_token_id] * self.w).to(config.device)
        self.cls_tokens = self.cls_tokens.unsqueeze(0).unsqueeze(-1).repeat(config.batch_size, 1, 1)
        self.cls_tokens.requires_grad = False

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
        self.seg = nn.Linear(config.d_model, self.output_size)

        # Auxiliary Regressor
        # self.aux = nn.Linear(config.d_model, 1)
        # self.aux_softmax = nn.Softmax(dim=2)
        # self.coherence_hinge_margin = config.coherence_hinge_margin
    
    def forward(self, x:torch.Tensor):
        b = x.size(0)
        w, s_ = self.w, self.s_

        # add bos token
        x = x.reshape(b, w, -1) # (b*w) x (s'-1)
        x = torch.cat([self.cls_tokens[:b], x], dim=-1) # (b*w) x s'
        x = x.reshape(b, w*s_) # b x (s+w)

        # wording embedding and positional encoding
        x = self.embedding(x) * self.embedding_scale # b x (s+w) x d
        x = x.reshape(b*w, s_, -1) # (b*w) x s' x d
        x += self.pe(x) # b x s' x d
        
        # encode each sentence
        masking = create_masking(s_, s_, x.device)
        for layer in self.sent_encoder:
            # (b*w) x s' x d
            x, _ = layer(x, masking=masking)
        y = x[:, -1, :] # (b*w) x d
        y = y.reshape(b, w, -1) # b x w x d

        # add positional encoding to windows
        y += self.pe(y) # b x w x d
        # encode each window (paragraph)
        for layer in self.window_encoder:
            # # b x w x d
            y, _ = layer(y)
        # b x w x 2
        y = self.seg(y)

        # calculate auxiliary loss
        # z = self.aux(y)
        # z = self.aux_softmax(z)
        # norm_scores_true = z[:, 0]
        # norm_scores_false = z[:, 1]
        # norm_scores = norm_scores_true - norm_scores_false
        # norm_scores = self.coherence_hinge_margin - norm_scores
        # aux_loss = torch.clamp(norm_scores, min=0)
        y = any_max_pooling(y)
        
        return y

