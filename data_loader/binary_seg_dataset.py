import glob
from os.path import join, exists

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.config import Config
from data_loader.setup_BPE import get_tokenizer


class BinarySegDataset(Dataset):
    """
    Classifies whether the current window has
    a segment boundary or not, this dataset needs tokenization
    """

    def __init__(self, config:Config, name):
        self.config = config
        self.files_path = join(config.data_path, name)
        self.seq_len = config.seq_len
        
        self.cache_path = join(config.data_path, 'cache',
            f'{name}_binary_{self.seq_len}.pt')

        self.segments = []
        self.name = name

    def load_data(self):
        
        if exists(self.cache_path):
            return
        
        if not self.config.is_host:
            return

        seg_files = glob.glob(join(self.files_path, '*.pt'))

        true_labels = 0
        np.random.shuffle(seg_files)
        print('Loading dataset')
        for seg_file in tqdm(seg_files):

            x, y = BinarySegDataset.build_file(seg_file)

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

        np.random.shuffle(self.segments)
        torch.save(self.segments, self.cache_path)
        print(f'Total number of {self.name} samples: {len(self.segments)}')

    @staticmethod
    def build_file(seg_file):
        tokenizer = get_tokenizer()
        segs = torch.load(seg_file)
        x = []
        y = []
        for seg, _ in segs:
            tokens = tokenizer.encode(seg, add_special_tokens=False)
            x.extend(tokens)
            y.extend([0] * (len(tokens) - 1))
            y.append(1)
        return x, y

    def load_cache(self):
        if len(self.segments) > 0:
            return
        assert exists(self.cache_path), 'cache not found'
        self.segments = torch.load(self.cache_path)
        if self.config.is_host:
            print(f'Total number of {self.name} samples: {len(self.segments)}')

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        tokens, label = self.segments[index]
        x = torch.tensor(tokens, dtype=torch.long)
        y = torch.tensor([label]).type(torch.long)
        return x, y
