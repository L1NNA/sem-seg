import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, BertConfig

from layers.pooling import cls_pooling
from layers.transformer import TransformerLayer, FFN
from layers.embeddings import PositionalEncoding
from layers.masking import create_segment_masking
from utils.config import Config
from utils.setup_BPE import get_model_path

def add_args(parser):
    parser.add_argument("--sim_loss", type=str, default="mse",
                        choices=("mse", "cosine", "s_mse", "s_cosine"),
                        help="Regression loss for clone search")
    parser.add_argument('--seg_masking', action='store_true',
                        help='if segmentaiton masking is applied')


class ChainOfExperts(nn.Module):

    def __init__(self, config:Config, output_dim):
        super(ChainOfExperts, self).__init__()
        self.w = config.n_windows
        self.output_size = output_dim
        self.sim_loss = config.sim_loss
        self.seg_masking = config.seg_masking
        model_path = get_model_path(config.bert_name)
        bert_config:BertConfig = AutoConfig.from_pretrained(model_path)
        self.encoder = AutoModel.from_pretrained(model_path,
                                                config=bert_config,
                                                add_pooling_layer=False)
        d = bert_config.hidden_size
        
        self.pe = PositionalEncoding(d)
        self.decoder = nn.ModuleList([
            TransformerLayer(d, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.w_layers)
        ])

        # Segmentation Classifier
        self.seg_output = nn.Linear(d, output_dim[0])
        # Labeling Classifier
        self.cls_output = nn.Linear(d, output_dim[1])
    
    def _fill_masking(self, attention_mask):
        if attention_mask is not None:
            seq_len = attention_mask.size(1)
            win_len = seq_len // self.w
            attention_mask[:, [i*win_len for i in range(self.w)]] = 1
        return attention_mask

    def _encode(self, x, attention_mask, b, w):
        # Pass inputs through GraphCodeBERT
        h = self.encoder(input_ids=x, attention_mask=attention_mask).last_hidden_state # (b*w) x s' x d
        h = cls_pooling(h) # (b*w) x d
        h = h.reshape(b, w, -1) # b x w x d
        return h

    def _decode(self, h, segs):
        h += self.pe(h) # b x w x d
        masking = None
        if self.seg_masking and segs is not None:
            seg_ids = torch.argmax(torch.softmax(segs, dim=1), dim=2) # b x w x 2
            masking = create_segment_masking(seg_ids)
        for layer in self.decoder:
            h, _ = layer(h, masking) # b x w x d
        return h

    def forward(self, x, attention_mask):
        b = x.size(0)
        b_ = b * self.w
        
        # add cls token
        attention_mask = self._fill_masking(attention_mask)
        x = x.reshape(b_, -1) # (b*w) x s'
        if attention_mask is not None:
            attention_mask = attention_mask.reshape(b_, -1)
        
        # encoding
        hx = self._encode(x, attention_mask, b, self.w) # b x w x d
        # Decoding
        px = self._decode(hx, None) # b x w x d
        px = px.reshape(b * self.w, -1)

        # Segmentation
        segs = self.seg_output(hx) # b x w x 2
        segs = segs.reshape(b * self.w, -1)
        # Labeling
        labeling = self.cls_output(px) # b x w x cls
        labeling = labeling.reshape(b * self.w, -1)

        return segs, labeling, px

    def embedding(self, x, attention_mask):
        b = x.size(0)
        b_ = b * self.w

        # add cls token
        attention_mask = self._fill_masking(attention_mask)
        x = x.reshape(b_, -1) # (b*w) x s'
        if attention_mask is not None:
            attention_mask = attention_mask.reshape(b_, -1)
        
        # encoding
        hx = self._encode(x, attention_mask, b, self.w) # b x w x d
        # Decoding
        px = self._decode(hx, None) # b x w x d
        px = px.reshape(b * self.w, -1)
        return px
