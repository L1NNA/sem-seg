import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaConfig, RobertaModel

from utils.setup_BPE import GRAPH_CODE_BERT, get_tokenizer
from layers.pooling import cls_pooling
from layers.transformer import TransformerLayer
from utils.config import Config

def add_args(parser):
    parser.add_argument("--sim_loss", type=str, default="mse",
                        choices=("mse", "s_mse", "s_cosine"),
                        help="Activation function for cosformer",
    )


class ChainOfExperts(nn.Module):

    def __init__(self, config:Config, num_classes):
        super(ChainOfExperts, self).__init__()
        self.w = config.n_windows
        assert config.seq_len % self.w == 0, \
            "seq_len must be divisible by n_windows"
        self.s_ = (config.seq_len // self.w) + 1
        self.cls_token = get_tokenizer().cls_token_id
        self.output_size = num_classes
        self.sim_loss = config.sim_loss

        bert_config:RobertaConfig = RobertaConfig.from_pretrained(GRAPH_CODE_BERT)
        self.encoder = RobertaModel.from_pretrained(GRAPH_CODE_BERT,
                                                    config=bert_config,
                                                    add_pooling_layer=False)
        d = bert_config.hidden_size

        # Segmentation Classifier
        self.seg_output = nn.Linear(d, 2)
        # Labeling Classifier
        self.cls_output = nn.Linear(d, num_classes)
        # Similarity Regression
        self.decoder = nn.ModuleList([
            TransformerLayer(d, bert_config.num_attention_heads, bert_config.intermediate_size, config.dropout)
            for _ in range(config.w_layers)
        ])
    
    def _fill_masking(self, attention_mask, b_, device):
        if attention_mask is not None:
            attention_mask = attention_mask.reshape(b_, -1)
            ones = torch.ones((b_, 1)).to(device)
            attention_mask = torch.cat([ones, attention_mask], dim=1)
        return attention_mask

    def _reshape_inputs(self, x, b_):
        x = x.reshape(b_, -1) # (b*w) x (s'-1)
        cls_tokens = torch.full((b_, 1), self.cls_token).to(x.device)
        x = torch.cat([cls_tokens, x], dim=1) # (b*w) x s'
        return x

    def _encode(self, x, attention_mask, b, w):
        # Pass inputs through GraphCodeBERT
        h = self.encoder(input_ids=x, attention_mask=attention_mask) # (b*w) x s' x d
        h = cls_pooling(h) # (b*w) x d
        h = h.reshape(b, w, -1) # b x w x d
        return h

    def _decode(self, h):
        for layer in self.decoder:
            h, _ = layer(h) # b x w x d
        return h

    def forward(self, x, attention_mask, y, y_attention_mask):
        b = x.size(0)
        w, s_ = self.w, self.s_
        b_ = b * w

        # add cls token
        x = self._reshape_inputs(x, b_) # (b*w) x s'
        attention_mask = self._fill_masking(attention_mask, b_)
        y = self._reshape_inputs(y, b_) # (b*w) x s'
        y_attention_mask = self._fill_masking(y_attention_mask, b_)
        
        # encoding
        hx = self._encode(x, attention_mask, b, w) # b x w x d
        hy = self._encode(y, y_attention_mask, b, w) # b x w x d
        # Decoding
        px = self._decode(hx)
        py = self._decode(hy)

        # Segmentation
        segs = self.seg_output(px) # b x w x 2
        # Labeling
        labeling = self.cls_output(px) # b x w x cls
        # similarity regression
        similarity = self.similarity_regression(hx, hy, px, py)

        return segs, labeling, similarity

    def similarity_regression(self, hx, hy, px, py):
        if self.sim_loss == 'mse':
            return F.mse_loss(px, py)
        return self._siamese_similarity_loss(px, hy) / 2 \
            + self._siamese_similarity_loss(py, hx) / 2

    def _siamese_similarity_loss(self, p, h):
        h = h.detach()
        if self.sim_loss == 's_mse':
            return F.mse_loss(p, h)
        else:
            return -F.cosine_similarity(p, h, dim=2).sum(dim=1).mean()
        


