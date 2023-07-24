import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel

from data_loader.setup_BPE import GRAPH_CODE_BERT
from layers.masking import create_masking
from utils.config import Config


def add_args(_):
    # vocab_size=50265
    # d_model=768
    # num_layers=12
    # num_heads=12
    # d_ff=3072
    pass

class GraphCodeBERT(nn.Module):

    def __init__(self, config:Config) -> None:
        super(GraphCodeBERT, self).__init__()
        self.config = config

        bert_config = RobertaConfig.from_pretrained(GRAPH_CODE_BERT)
        self.encoder = RobertaModel.from_pretrained(GRAPH_CODE_BERT,
                                                    config=bert_config)

        self.output = nn.Linear(bert_config.hidden_size, config.vocab_size)

    def forward(self, x:torch.Tensor):
        attn_mask = create_masking(x.size(1), x.size(1))
        # start from 1 since 0 is for bos token
        position_idx = torch.arange(1, x.size(1) + 1,
                                    dtype=torch.long, device=x.device)
        y = self.encoder(x,
                         attention_mask=attn_mask,
                         position_ids=position_idx)
        y = self.output(y)
        return y