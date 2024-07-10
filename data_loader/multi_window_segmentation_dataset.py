import glob
from os.path import join, exists
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.config import Config
from utils.setup_BPE import get_tokenizer
from utils.dist_dataset import DistDataset


class MultiWindowSegmentationDataset(DistDataset):
    """
    Classifies whether the current window with adjacent windows has
    a segment boundary or not.
    """

    def __init__(self, config:Config, stage):
        super().__init__(config, f'{config.data}_{config.seq_len}_{config.n_windows}', stage)
        self.n_windows = config.n_windows
        
    def load_data(self):
        
        if exists(self.indices_cache_path):
            return

        print('Loading dataset...')
        true_labels, false_labels = 0, 0
        for index, seg_file in enumerate(self.tokens):
            seg_file:List[Tuple[List[str], str]]
            get_labels = lambda tokens:[0] * (len(tokens)-1) + [1]
            all_labels:List[int] = [label for tokens,_ in seg_file for label in get_labels(tokens)]

            window_size = self.seq_len // self.n_windows
            for i in range(0, len(all_labels)-self.seq_len, self.seq_len):
                labels = [1 if 1 in all_labels[i+j*window_size:i+(j+1)*window_size] else 0 for j in range(self.n_windows)]
                self.indices.append((index, i, labels))

        torch.save(self.indices, self.indices_cache_path)

    def __getitem__(self, index):
        i, j, labels = self.indices[index]
        tokens = []
        for token,_,_ in self.get_all(i, j, self.seq_len):
            tokens.append(token)
        x = torch.tensor(tokens, dtype=torch.long)
        y = torch.tensor(labels, dtype=torch.long)
        return x, y
