import glob
from os.path import join, basename, exists

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.config import Config


class BinaryDataset(Dataset):
    """
    Classifies whether the current window has
    a segment boundary or not 
    """

    def __init__(self, config:Config, name, max_len=100000):
        self.config = config
        self.token_path = join(config.data_path, name)
        self.seq_len = config.seq_len
        
        self.max_len = max_len
        self.cache_path = join(config.data_path, 'cache',
            f'{name}_binary_{self.seq_len}.pt')

        self.segments = []

    def load_data(self):
        
        if exists(self.cache_path):
            return
        
        if not self.config.is_host:
            return

        x_files = glob.glob(join(self.token_path, '*.x.pt'))
        y_files = glob.glob(join(self.token_path, '*.y.pt'))
        y_map = {basename(f)[:-5]:f for f in y_files}

        true_labels = 0
        for x_file in x_files:
            x_name = basename(x_file)[:-5]
            y_file = y_map[x_name]

            x = torch.load(x_file)
            y = torch.load(y_file)

            if len(x) < self.seq_len:
                continue

            false_labels = []
            # reverse to prevent padding
            for i in range(len(x)-self.seq_len, -1, -self.seq_len):
                if 1 in y[i:i+self.seq_len]:
                    tokens = x[i:i+self.seq_len]
                    self.segments.append((tokens, 1))
                    true_labels += 1
                else:
                    false_labels.append(i)
            if len(false_labels) > true_labels:
                false_labels = np.random.choice(false_labels, true_labels)
            true_labels -= len(false_labels)

            for i in false_labels:
                tokens = x[i:i+self.seq_len]
                self.segments.append((tokens, 0))

            if len(self.segments) > self.max_len:
                break
        np.random.shuffle(self.segments)
        torch.save(self.segments, self.cache_path)

    def load_cache(self):
        assert exists(self.cache_path), 'cache not found'
        if len(self.segments) > 0:
            return
        self.segments = torch.load(self.cache_path)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        tokens, label = self.segments[index]
        x = torch.tensor(tokens, dtype=torch.long)
        y = torch.tensor([label]).type(torch.long)
        return x, y
