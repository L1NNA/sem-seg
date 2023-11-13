import torch
import torch.nn as nn
import math

from layers.embeddings import PositionalEncoding
from layers.masking import create_masking
from layers.pooling import next_token_pooling
from layers.transformer import TransformerLayer
from utils.config import Config


def add_args(parser):
    parser.add_argument(
        "--d_model", type=int, default=512, help="hidden dimension"
    )
    parser.add_argument(
        "--n_heads", type=int, default=4, help="number of attention heads"
    )
    parser.add_argument(
        "--n_layers", type=int, default=6, help="number of layers"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="dropout rate"
    )
    parser.add_argument(
        "--d_ff", type=int, default=1024, help="feed forward hidden dimension"
    )


class Transformer(nn.Module):

    def __init__(self, args:Config, output_dim) -> None:
        super(Transformer, self).__init__()

        self.n_heads = args.n_heads
        self.n_layers = args.n_layers
        self.d_model = args.d_model
        self.dropout = args.dropout
        self.d_ff = args.d_ff
        self.vocab_size = args.vocab_size
        self.output_size = output_dim
        
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.embedding_scale = math.sqrt(self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model)
        self.layers = nn.ModuleList([
            TransformerLayer(self.d_model, self.n_heads, self.d_ff, self.dropout)
            for _ in range(self.n_layers)
        ])
        self.output = nn.Linear(self.d_model, self.output_size)


    def forward(self, x:torch.Tensor):
        x = self.embedding(x) * self.embedding_scale
        x += self.pos_encoding(x)
        seq_len = x.size(1)
        masking = create_masking(seq_len, seq_len, x.device)
        attns = {}
        for i, layer in enumerate(self.layers):
            x, attn = layer(x, masking=masking)
            attns[f'attn_{i}'] = attn
        output = self.output(x)
        output = next_token_pooling(output)
        return output