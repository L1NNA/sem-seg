import torch
import torch.nn as nn


class StatefulTransformer(nn.Module):
    """
    Transformer with states
    """
    
    def __init__(self, vocab_size=5000, embedding_dim=300, n_head=4, e_layers=6):
        d_model = embedding_dim

        # embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, e_layers)

        # States
        self.stateful = nn.Linear(d_model, d_model)
        self.seg = nn.Linear(d_model, 2)
        self.seg_softmax = nn.Softmax(dim=2)

    
    def forward(self, x, state):
        b, s = x.shape

        x = self.embedding(x) + state

        # encode
        y = self.encoder(x)

        # state
        state = self.stateful(y)

        # prediction
        z = self.seg(y)
        z = self.seg_softmax(z)
        
        return z, state

