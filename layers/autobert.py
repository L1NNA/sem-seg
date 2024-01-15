import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from layers.pooling import cls_pooling
from utils.config import Config
from utils.setup_BPE import get_tokenizer, get_model_path


def add_args(parser):
    parser.add_argument("--bert_name", default=None,
                        choices=['graphcodebert', 'longformer', 'sentbert'],
                        help="Path to the pre-trained model",
    )

class AutoBERT(nn.Module):

    def __init__(self, config:Config, output_dim) -> None:
        super(AutoBERT, self).__init__()
        self.config = config
        self.output_size = output_dim
        model_path = get_model_path(config.bert_name)
        self.cls_token_id = get_tokenizer().cls_token_id
        bert_config = AutoConfig.from_pretrained(model_path)
        self.encoder = AutoModel.from_pretrained(model_path,
                                                config=bert_config,
                                                add_pooling_layer=False)
        self.output = nn.Linear(bert_config.hidden_size, self.output_size)

    def forward(self, x):
        # add cls token at the beginning
        b = x.size(0)
        cls_tokens = torch.full((b, 1), self.cls_token_id).to(x.device)
        x = torch.cat([cls_tokens, x], dim=1)
        # encoding
        h = self.encoder(x).last_hidden_state # b x l x d
        # pooling
        h = cls_pooling(h) # b x d
        output = self.output(h)
        return output