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
    """

    def __init__(self, config:Config, output_dim) -> None:
        super(GraphCodeBERT, self).__init__()
        self.config = config
        self.output_size = output_dim
        
        self.cls_tokens = torch.tensor([get_tokenizer().bos_token_id]).to(config.device)
        self.cls_tokens = self.cls_tokens.repeat(config.batch_size, 1)
        self.cls_tokens.requires_grad = False

        bert_config = RobertaConfig.from_pretrained(GRAPH_CODE_BERT)
        self.encoder = RobertaModel.from_pretrained(GRAPH_CODE_BERT,
                                                    config=bert_config,
                                                    add_pooling_layer=False)

        self.output = nn.Linear(bert_config.hidden_size, self.output_size)

    def forward(self, x:torch.Tensor):
        # start from 1 since 0 is for bos token
        x = torch.cat([self.cls_tokens[:x.size(0)], x], dim=1)
        position_idx = torch.arange(x.size(1), dtype=torch.long, device=x.device)
        y = self.encoder(x, position_ids=position_idx).last_hidden_state
        y = self.output(y)
        y = cls_pooling(y)
        return y