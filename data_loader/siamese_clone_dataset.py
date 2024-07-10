import glob
from os.path import join, exists
from typing import List, Tuple

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from utils.config import Config
from utils.setup_BPE import get_tokenizer
from utils.dist_dataset import DistDataset
from utils.label_utils import get_seg_type, SegType


class SiameseCloneDataset(DistDataset):

    def __init__(self, config:Config, stage):
        super().__init__(config, f'{config.data}_{config.seq_len}_{config.n_windows}_{config.skip_label}', stage)
        self.window_size = config.seq_len // config.n_windows

    def __getitem__(self, index):
        i, j, k, l, label = self.indices[index]
        if label == 0 and self.stage == 'train':
            label = -1
        source_tokens = []
        dest_tokens = []
        for token, _, _ in self.get_all(i, j, self.seq_len):
            source_tokens.append(token)
        for token, _, _ in self.get_all(k, l, self.seq_len):
            dest_tokens.append(token)

        x = torch.tensor(source_tokens, dtype=torch.long)
        y = torch.tensor(dest_tokens, dtype=torch.long)
        z = torch.tensor([label], dtype=torch.long)

        return x, y, z

    def load_data(self):
        if exists(self.indices_cache_path):
            return
        
        print('Loading dataset...')
        for i, seg_file in enumerate(self.tokens):
            j = 0
            seg_file:List[Tuple[List[str], str]]
            for tokens, seg_name in seg_file:
                seg_type = get_seg_type(seg_name)
                remainder = j % self.window_size
                if seg_type >= self.config.skip_label \
                  and remainder + len(tokens) >= self.seq_len \
                  and len(tokens) > self.seq_len:
                    self.indices.append((i, j-remainder, i, j, 1))
                j += len(tokens)
        shuffled = SiameseCloneDataset.sattolo_cycle(self.indices)
        self.indices.extend(shuffled)
        torch.save(self.indices, self.indices_cache_path)

    @staticmethod    
    def sattolo_cycle(indices):
        shuffled = indices.copy()
        n = len(indices)
        for last in range(n - 1, 0, -1):
            rand = np.random.randint(last)
            last_i, last_j, last_k, last_l, _ = shuffled[last]
            rand_i, rand_j, rand_k, rand_l, _ = shuffled[rand]
            shuffled[last] = (last_i, last_j, rand_k, rand_l, 0)
            shuffled[rand] = (rand_i, rand_j, last_k, last_l, 0)
        return shuffled
