import glob
from os.path import join, exists

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.distributed as dist

from utils.config import Config
from utils.setup_BPE import get_tokenizer
from utils.dist_dataset import DistDataset
from utils.label_utils import label_seg, get_seg_type, SegType


class SingleLabelingDataset(DistDataset):
    """
    Classifies the type of the current window
    """

    def __init__(self, config:Config, stage):
        super().__init__(config, f'{config.data}_{config.seq_len}_{config.skip_label}', stage)
        self.skip_label = config.skip_label
        self.pad_token_id = get_tokenizer().pad_token_id

    def __getitem__(self, index):
        i, j, k = self.indices[index]

        tokens = self.tokens[i][j][0][:self.seq_len]
        paddings = [self.pad_token_id] * (self.seq_len - len(tokens))
        x = torch.tensor(tokens + paddings, dtype=torch.long)
        masking = torch.ones(len(tokens), dtype=torch.long)
        if len(paddings) > 0:
            masking = torch.concat([masking, torch.zeros(len(paddings), dtype=torch.long)])
        y = torch.tensor([k], dtype=torch.long)
        return x, masking, y

    def pipeline(self):
        # tokens
        if self.config.is_host:
            self.cache_tokens()
        if self.config.distributed:
            dist.barrier()
        self.load_tokens()
        if self.config.distributed:
            dist.barrier()

        for i, segs in enumerate(self.tokens):
            for j, (token, label) in enumerate(segs):
                label_type = get_seg_type(label).value
                if self.skip_label > label_type:
                    continue
                label_type -= self.skip_label
                self.indices.append((i, j, label_type))

        if self.config.is_host:
            print(f'Total number of {self.stage} samples: {len(self)}')