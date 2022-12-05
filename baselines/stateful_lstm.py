import torch
import torch.nn as nn

from data_loader.tokenizer import bert_tokenizer


class StatefulLSTM(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        vocab = len(bert_tokenizer.get_vocab())
        self.embedding = nn.Embedding(vocab, 512)
        self.lstm = nn.LSTM(512, 1024, 2, batch_first=True)

    def forward(self, x):
        y = self.embedding(x)
        return self.lstm(y)

