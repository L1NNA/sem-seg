import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel

from utils.setup_BPE import GRAPH_CODE_BERT, get_tokenizer
from layers.pooling import cls_pooling
from utils.config import Config


class BERT_R_CNN(nn.Module):
    def __init__(self, config:Config, num_classes, num_boundaries=2):
        super(BERT_R_CNN, self).__init__()
        bert_config = RobertaConfig.from_pretrained(GRAPH_CODE_BERT)
        self.encoder = RobertaModel.from_pretrained(GRAPH_CODE_BERT,
                                                    config=bert_config,
                                                    add_pooling_layer=False)

        # Example: Two-level Conv1D for R-CNN part
        hidden_size = bert_config.hidden_size
        self.conv1 = nn.Conv1d(in_channels=hidden_size, out_channels=config.d_ff, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=config.d_ff, out_channels=hidden_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc_class = nn.Linear(hidden_size, num_classes)
        # self.fc_boundary = nn.Linear(hidden_size, num_boundaries)

    def forward(self, x:torch.Tensor, attention_mask=None):
        # Pass inputs through GraphCodeBERT
        h = self.encoder(input_ids=x, attention_mask=attention_mask)
        h = h.last_hidden_state.transpose(1, 2)  # Transpose for Conv1D

        # Pass through Conv1D layers
        h = self.relu(self.conv1(h))
        h = self.relu(self.conv2(h))
        h = h.transpose(1, 2)

        # Classification and boundary prediction
        z = cls_pooling(h)
        class_logits = self.fc_class(z)
        # boundary_logits = self.fc_boundary(z)

        return class_logits
