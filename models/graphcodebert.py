"""
GraphCodeBERT: Pre-training Code Representations with Data Flow
https://arxiv.org/abs/2009.08366
"""
import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel

from utils.setup_BPE import GRAPH_CODE_BERT, get_tokenizer
from layers.pooling import cls_pooling
from utils.config import Config


def add_args(_):
    pass

class GraphCodeBERT(nn.Module):
    """
    the hyperparameters are fixed as the followings:
    vocab_size=50265
    d_model=768
    num_layers=12
    num_heads=12
    d_ff=3072
    seq_len=512
    """

    def __init__(self, config:Config, output_dim) -> None:
        super(GraphCodeBERT, self).__init__()
        self.config = config
        self.output_size = output_dim
        self.cls_token_id = get_tokenizer().cls_token_id

        bert_config = RobertaConfig.from_pretrained(GRAPH_CODE_BERT)
        self.encoder = RobertaModel.from_pretrained(GRAPH_CODE_BERT,
                                                    config=bert_config,
                                                    add_pooling_layer=False)
        if self.output_size > 0:
            self.output = nn.Linear(bert_config.hidden_size, self.output_size)

    def forward(self, x:torch.Tensor, masking:torch.Tensor):
    
        y = self.encoder(x, attention_mask=masking).last_hidden_state
        y = cls_pooling(y)
        if self.output_size > 0:
            y = self.output(y)
        return y