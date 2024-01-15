import glob
from os.path import join, exists

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

from utils.config import Config
from utils.setup_BPE import get_tokenizer
from utils.dist_dataset import DistDataset
from labeling.data_loader.label_utils import label_seg, get_seg_type


class LabelingDataset(DistDataset):
    """
    Classifies the type of the current window
    """

    def __init__(self, config:Config, stage):
        super().__init__(config, f'{config.data}_{config.seq_len}', stage)
        self.config = config
        self.segments = []
        self.seq_len = config.seq_len
        self.files_path = join(config.data_path, stage)
        self.cache_path = join(config.data_path, 'cache',  config.data, f'{self.name}.pt')

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        tokens, label = self.segments[index]
        x = torch.tensor(tokens, dtype=torch.long)
        y = torch.tensor([label], dtype=torch.long)
        return x, y

    def load_data(self):
        
        if exists(self.cache_path):
            return

        print('Loading dataset')
        for seg_file in tqdm(glob.glob(join(self.files_path, '*.pt'))):

            x, y = LabelingDataset.build_file(seg_file)

            if len(x) < self.seq_len:
                continue

            indices = [*range(0, len(x)-self.seq_len, self.seq_len)]
            if len(indices) > 100:
                indices = np.random.choice(indices, 100, replace=False)

            for i in indices:
                tokens = x[i:i+self.seq_len]
                labels = y[i:i+self.seq_len]
                label = label_seg(labels)
                self.segments.append((tokens, label))

        torch.save(self.segments, self.cache_path)
        print(f'Total number of {self.stage} samples: {len(self.segments)}')

    @staticmethod
    def build_file(seg_file):
        tokenizer = get_tokenizer()
        segs = torch.load(seg_file)
        x = []
        y = []
        for seg, label in segs:
            tokens = tokenizer.encode(seg, add_special_tokens=False)
            x.extend(tokens)
            y.extend([label] * len(tokens))
        return x, y

    def load_cache(self):
        if len(self.segments) > 0:
            return
        assert exists(self.cache_path), 'cache not found'
        self.segments = torch.load(self.cache_path)
        if self.config.is_host:
            print(f'Total number of {self.stage} samples: {len(self.segments)}')