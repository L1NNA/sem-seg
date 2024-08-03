"""
https://sbert.net/
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from layers.pooling import cls_pooling
from utils.config import Config
from utils.setup_BPE import get_model_path


def add_args(parser):
    parser.add_argument("--bert_name", default='sentbert',
                        help="Path to the pre-trained model",
    )

class SentBERT(nn.Module):

    def __init__(self, config:Config, output_dim) -> None:
        super(SentBERT, self).__init__()
        self.config = config
        self.output_size = output_dim
        model_path = get_model_path(config.bert_name)
        bert_config = AutoConfig.from_pretrained(model_path)
        self.encoder = AutoModel.from_pretrained(model_path,
                                                config=bert_config,
                                                add_pooling_layer=False)
        if self.output_size > 0:
            self.output = nn.Linear(bert_config.hidden_size, self.output_size)

    def forward(self, x, masking=None):
        # encoding
        h = self.encoder(x, attention_mask=masking).last_hidden_state # b x l x d
        # pooling
        y = cls_pooling(h) # b x d
        if self.output_size > 0:
            y = self.output(y)
        return y