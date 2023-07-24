"""
Two-Level Transformer and Auxiliary Coherence
Modeling for Improved Text Segmentation
"""
import math

import torch
import torch.nn as nn

from layers.embeddings import PositionalEncoding
from utils.config import Config


def add_args(args):
    args.add_argument("--w_layers", type=int, default=6,
                      help="number of layers for window encoder")
    args.add_argument("--coherence_hinge_margin", type=float, default=0)


class CATS(nn.Module):
    
    def __init__(self, config:Config, pos_dim=10):
        super(CATS, self).__init__()
        self.pos_dim = pos_dim

        # embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model) 
        self.pe = PositionalEncoding(config.d_model)
        self.embedding_scale = math.sqrt(config.d_model)

        # Encode each sentence
        self.sent_encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.n_heads, batch_first=True)
        self.sent_encoder = nn.TransformerEncoder(
            self.sent_encoder_layer, config.n_layers)

        # Encode the whole window (paragraph)
        self.window_encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.n_heads, batch_first=True)
        self.window_encoder = nn.TransformerEncoder(
            self.window_encoder_layer, config.w_layers)

        # Segmentation Classifier
        self.seg = nn.Linear(config.d_model, 2)
        self.seg_softmax = nn.Softmax(dim=2)

        # TODO: Auxiliary Regressor
        self.aux = nn.Linear(config.d_model, 1)
        self.aux_softmax = nn.Softmax(dim=2)
        self.coherence_hinge_margin = config.coherence_hinge_margin
    
    def forward(self, x:torch.Tensor):
        b, w, s = x.shape

        # wording embedding and positional embedding
        window_pos = self.pe(x).reshape(w, 1, self.pos_dim).repeat(b, s, 1)
        x = x.resahpe(b*w, s)
        sent_pos = self.pe(x).repeat([b*w, 1, 1])
        x = self.embedding(x) * self.embedding_scale
        # CATS does concatenation instead of summation for positional embedding
        x = torch.cat((x, sent_pos, window_pos), dim=-1)

        # encode each sentence
        y = self.sent_encoder(x)
        y = y[:, 0, :]
        y = y.reshape(b, w, -1)

        # encode each window (paragraph)
        y = self.window_encoder(y)
        y = self.seg(z)
        y = self.seg_softmax(z)

        z = self.aux(y)
        z = self.aux_softmax(z)

        
        norm_scores_true = z[:, 0]
        norm_scores_false = z[:, 1]
        norm_scores = norm_scores_true - norm_scores_false
        norm_scores = self.coherence_hinge_margin - norm_scores
        aux_loss = torch.clamp(norm_scores, min=0)
        return z, aux_loss

