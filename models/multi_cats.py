"""
Two-Level Transformer and Auxiliary Coherence
Modeling for Improved Text Segmentation
https://arxiv.org/abs/2001.00891
"""
import math

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, BertConfig

from utils.setup_BPE import GRAPH_CODE_BERT, get_tokenizer, get_model_path
from layers.transformer import TransformerLayer
from layers.embeddings import PositionalEncoding
from layers.pooling import any_max_pooling, cls_pooling
from utils.config import Config


class MultiCATS(nn.Module):
    
    def __init__(self, config:Config, output_dim):
        super(MultiCATS, self).__init__()
        self.config = config
        self.w = config.n_windows
        self.output_size = output_dim
        assert config.seq_len % self.w == 0, \
            "seq_len must be divisible by n_windows"
        self.cls_token = get_tokenizer().cls_token_id

        model_path = get_model_path(config.bert_name)
        bert_config:BertConfig = AutoConfig.from_pretrained(model_path)
        self.encoder = AutoModel.from_pretrained(model_path,
                                                config=bert_config,
                                                add_pooling_layer=False)
        d = bert_config.hidden_size

        # Similarity Regression
        self.decoder = nn.ModuleList([
            TransformerLayer(d, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.w_layers)
        ])

        # Segmentation Classifier
        self.seg = nn.Linear(d, self.output_size)

    def _reshape_inputs(self, x, b_):
        x = x.reshape(b_, -1) # (b*w) x (s'-1)
        cls_tokens = torch.full((b_, 1), self.cls_token).to(x.device)
        x = torch.cat([cls_tokens, x], dim=1) # (b*w) x s'
        return x
    
    def forward(self, x:torch.Tensor):
        b = x.size(0)
        w = self.w
        b_ = b * w

        # add cls token
        x = self._reshape_inputs(x, b_) # (b*w) x s'
        # Encode each window
        hx = self.encoder(x).last_hidden_state
        hx = cls_pooling(hx) # (b*w) x d
        hx = hx.reshape(b, w, -1) # b x w x d
        
        # decode global window
        for layer in self.decoder:
            # b x w x d
            hx, _ = layer(hx)

        output = self.seg(hx) # b x w x 2
        output = output.reshape(b_, -1) # (b*w) x 2
        return output

