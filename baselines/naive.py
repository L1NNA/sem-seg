import torch
import torch.nn as nn

from data_loader.tokenizer import bert_tokenizer


class Naive(nn.Module):

    def __init__(self) -> None:
       super().__init__()
       self.export_id = bert_tokenizer.convert_tokens_to_ids(['exports'])[0]

    def forward(self, x):
        y = (x == self.export_id).bool()
        y = torch.reshape(y, (-1,))
        y = torch.any(y, 1).int()
        return y


