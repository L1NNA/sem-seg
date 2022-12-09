import torch
import torch.nn as nn

from data_loader.tokenizer import bert_tokenizer
from utils.config import Config


class StatefulLSTM(nn.Module):

    def __init__(self, config:Config) -> None:
        super().__init__()

        self.input_size = config.seq_len
        self.d_model = config.d_model
        self.hidden_size = config.d_ff
        self.num_layers = config.e_layers
        vocab = len(bert_tokenizer.get_vocab())

        self.embedding = nn.Embedding(vocab, self.d_model)
        self.lstm = nn.LSTM(self.d_model, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, states):
        y = self.embedding(x)
        out, states = self.lstm(y, states)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out, states

