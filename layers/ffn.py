import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):

    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(FFN, self).__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x