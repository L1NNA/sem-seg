import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from layers.pooling import cls_pooling
from utils.config import Config
from utils.setup_BPE import get_tokenizer, get_model_path


def add_args(parser):
    parser.add_argument("--bert_name", default=None,
                        choices=['graphcodebert', 'sentbert'],
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
        if self.output_size > 0:
            self.output = nn.Linear(bert_config.hidden_size, self.output_size)

    def forward(self, x, masking=None):
        # add cls token at the beginning
        x = self._add_tokens(x, self.cls_token_id)
        if masking is not None:
            masking = self._add_tokens(masking, 1)
        # encoding
        h = self.encoder(x, attention_mask=masking).last_hidden_state # b x l x d
        # pooling
        y = cls_pooling(h) # b x d
        if self.output_size > 0:
            y = self.output(y)
        return y

    def _add_tokens(self, x, value):
        b = x.size(0)
        if self.config.n_windows > 1:
            b *= self.config.n_windows
            x = x.reshape(b, -1)
        tokens = torch.full((b, 1), value).to(x.device)
        x = torch.cat([tokens, x], dim=1)
        if self.config.n_windows > 1:
            b = b // self.config.n_windows
            x = x.reshape(b, -1)
        return x