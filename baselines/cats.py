import torch
import torch.nn as nn

from layers.embeddings import PositionalEmbedding


class CATS(nn.Module):
    """
    Two-Level Transformer and Auxiliary Coherence Modeling for Improved Text Segmentation
    """
    
    def __init__(self, vocab_size=5000, embedding_dim=300, pos_dim=10, n_head=4, s_layers=6, w_layers=6):
        d_model = embedding_dim + pos_dim * 2
        self.pos_dim = pos_dim

        # embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # TODO: Load FastText weights
        self.pe = PositionalEmbedding(self.pos_dim)

        # Encode each sentence
        self.sent_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.sent_encoder = nn.TransformerEncoder(self.sent_encoder_layer, s_layers)

        # Encode the whole window (paragraph)
        self.window_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.window_encoder = nn.TransformerEncoder(self.window_encoder_layer, w_layers)

        # Segmentation Classifier
        self.seg = nn.Linear(d_model, 2)
        self.seg_softmax = nn.Softmax(dim=2)

        # TODO: Auxiliary Regressor
        # self.aux = nn.Linear(d_model, 1)
        # self.aux_softmax = nn.Softmax(dim=2)

    
    def forward(self, x):
        b, w, s = x.shape

        # wording embedding and positional embedding
        window_pos = self.pe(x).reshape(w, 1, self.pos_dim).repeat(b, s, 1)
        x = x.resahpe(b*w, s)
        sent_pos = self.pe(x).repeat([b*w, 1, 1])
        x = self.embedding(x)
        # CATS does concatenation instead of summation for positional embedding
        x = torch.cat((x, sent_pos, window_pos), dim=-1)

        # encode each sentence
        y = self.sent_encoder(x)
        y = y[:, 0, :]
        y = y.reshape(b, w, -1)

        # encode each window (paragraph)
        z = self.window_encoder(y)
        z = self.seg(z)
        z = self.softmax(z)
        return z

