"""
Two-Level Transformer and Auxiliary Coherence
Modeling for Improved Text Segmentation
https://arxiv.org/abs/2001.00891
"""
import math

import torch
import torch.nn as nn

from utils.config import Config
from data_loader.setup_BPE import get_tokenizer


def add_args(args):
    args.add_argument("--n_windows", type=int, default=8,
                      help="number of windows (must divide seq_len)")
    args.add_argument("--w_layers", type=int, default=6,
                      help="number of layers for window encoder")
    # args.add_argument("--coherence_hinge_margin", type=float, default=0)


class CATS(nn.Module):
    
    def __init__(self, config:Config):
        super(CATS, self).__init__()
        self.config = config
        self.output_size = config.vocab_size if config.data == 'seq' else 2
        self.w = config.n_windows
        assert config.seq_len % self.w == 0, "seq_len must be divisible by n_windows"

        self.s_ = config.seq_len // self.w + 1
        self.start_tokens = torch.tensor(
            [get_tokenizer().bos_token_id], dtype=torch.long, device=config.device) \
            .repeat(config.batch_size, self.w, 1)

        # embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model) 
        self.pe = nn.Embedding(self.s_, config.d_model)
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
        self.seg = nn.Linear(config.d_model, self.output_size)

        # # TODO: Auxiliary Regressor
        # self.aux = nn.Linear(config.d_model, 1)
        # self.aux_softmax = nn.Softmax(dim=2)
        # self.coherence_hinge_margin = config.coherence_hinge_margin
    
    def forward(self, x:torch.Tensor):
        b = x.size(0)
        w, s_ = self.w, self.s_

        # add start tokens
        x = x.reshape(b, w, -1) # b x w x (s'-1)
        x = torch.cat((self.start_tokens, x), dim=-1) # b x w x s'
        x = x.reshape(b, -1) # b x (w*s')

        # wording embedding and positional encoding
        x = self.embedding(x) * self.embedding_scale # b x (w*s') x d
        x = x.reshape(b * w, s_, -1) # (b*w) x s' x d
        # b x (w*s') x d'
        sent_pos = self.pe(torch.arange(s_, device=x.device)).repeat([b*w, 1, 1])
        x += sent_pos # (b*w) x s' x d
        

        # encode each sentence
        y = self.sent_encoder(x, is_causal=True) # (b*w) x s' x d
        y = y[:, 0, :] # (b*w) x d
        y = y.reshape(b, w, -1) # b x w x d

        # encode each window (paragraph)
        y = self.window_encoder(y)
        y = self.seg(y)

        # calculate auxiliary loss
        # z = self.aux(y)
        # z = self.aux_softmax(z)
        # norm_scores_true = z[:, 0]
        # norm_scores_false = z[:, 1]
        # norm_scores = norm_scores_true - norm_scores_false
        # norm_scores = self.coherence_hinge_margin - norm_scores
        # aux_loss = torch.clamp(norm_scores, min=0)
        
        return y

