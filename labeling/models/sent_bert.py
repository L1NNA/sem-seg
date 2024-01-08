import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from layers.pooling import cls_pooling
from utils.config import Config


def add_args(_):
    pass

class SentenceBERT(nn.Module):

    def __init__(self, config:Config, output_dim) -> None:
        super(SentenceBERT, self).__init__()
        self.config = config
        self.output_size = output_dim
        self.encoder = RobertaModel.from_pretrained('sentence-transformers/all-MiniLM-L12-v2',
                                                    add_pooling_layer=False)

    def forward(self, inputs):
        x, y = inputs # b x l x d
        x = self.encoder(x).last_hidden_state # b x l x d
        y = self.encoder(y).last_hidden_state

        # pooling
        x = cls_pooling(x) # b x d
        y = cls_pooling(y) # b x d

        output = (F.cosine_similarity(x, y, dim=1) + 1) / 2
        return output