import glob
from os.path import join, exists

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

from utils.config import Config
from utils.setup_BPE import get_tokenizer
from utils.dist_dataset import DistDataset
from utils.label_utils import label_seg, get_seg_type, SegType


class LabelingDataset(DistDataset):
    """
    Classifies the type of the current window
    """

    def __init__(self, config:Config, stage):
        skip = '' if config.skip_label <= 0 else '_' + str(config.skip_label)
        super().__init__(config, f'{config.data}_{config.seq_len}{skip}', stage)

    def __getitem__(self, index):
        i, j, k = self.indices[index]
        seg_file = self.tokens[i]
        tokens, label = seg_file[j]
        label_type = get_seg_type(label).value - self.config.skip_label

        x = torch.tensor(tokens[k:k+self.seq_len], dtype=torch.long)
        y = torch.tensor([label_type], dtype=torch.long)
        return x, y

    def load_data(self):
        if exists(self.indices_cache_path):
            return

        print('Loading dataset...')

        # # 1.sliding window with no seg with padding
        # remainder = 0
        # for i, seg_file in enumerate(self.tokens):
        #     seg_file:List[Tuple[List[str], str]]
        #     for j, (tokens, _) in enumerate(seg_file):
        #         length = len(tokens)

        #         remainder = self.seq_len - remainder
        #         if length < remainder:
        #             remainder = self.seq_len - remainder + length
        #             continue
        #         if length < remainder + self.seq_len:
        #             remainder = length - remainder
        #             continue
        #         for k in range(remainder, length-self.seq_len+1, self.seq_len):
        #             self.indices.append((i, j, k))
        #         remainder = (length - remainder) % self.seq_len

        # 2.sliding window with no seg starting from zero
        for i, seg_file in enumerate(self.tokens):
            seg_file:List[Tuple[List[str], str]]
            for j, (tokens, label) in enumerate(seg_file):
                if get_seg_type(label).value >= self.config.skip_label and len(tokens) >= self.seq_len:
                    self.indices.append((i, j, 0))

        torch.save(self.indices, self.indices_cache_path)