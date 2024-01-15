import glob
from os.path import join, exists

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

from utils.config import Config
from utils.setup_BPE import get_tokenizer


def get_random_index(span, indices):
    possible_indices = np.setdiff1d(np.arange(span), indices, assume_unique=False)
    return np.random.choice(possible_indices)


class BinaryClassification(Dataset):

    def __init__(self, config:Config, name):
        range_len = 500
        self.files_path = join(config.data_path, name)
        self.token_cache = join(config.data_path, 'cache',
            f'{name}_binary_classification_tokens.pt')
        self.indices_cache = join(config.data_path, 'cache',
            f'{name}_binary_classification_{range_len}_indices.pt')
        self.range_len = range_len
        self.max_len = config.seq_len
        self.seq_len = (self.max_len - 2) // 2
        self.seg_tokens = []
        self.indices = []
        self.name = name
        self.limit = 8000
        self.tokenizer = get_tokenizer()
        self.config = config

    def load_cache(self):
        if self.config.is_host:
            return
        self.seg_tokens = torch.load(self.token_cache)
        self.indices = torch.load(self.indices_cache)

    def load_data(self):
        if not self.config.is_host:
            return
        if exists(self.token_cache):
            self.seg_tokens = torch.load(self.token_cache)
            print(self.name, 'tokens loaded')
        else:
            seg_files = glob.glob(join(self.files_path, '*.pt'))
            # load files and tokenization
            for seg_file in tqdm(seg_files, desc='Load files'):
                segs = torch.load(seg_file)
                for seg, _ in segs:
                    tokens = self.tokenizer.encode(seg, add_special_tokens=False)
                    if len(tokens) < self.seq_len:
                        continue
                    self.seg_tokens.append(tokens)
            torch.save(self.seg_tokens, self.token_cache)
        
        if exists(self.indices_cache):
            self.indices = torch.load(self.indices_cache)
            print(self.name, 'indices loaded')
        else:
            span = len(self.seg_tokens)
            if self.limit < span-1:
                indices = np.random.choice(range(1, span-1), self.limit, replace=False)
            else:
                indices = [*range(1, span-1)]
            for i in tqdm(indices, desc='Load indices'):
                # select another segmentation, such that (i, j) has label 0
                j = get_random_index(span, [i-1, i])
                # select some tokens from previous segment
                prev_index = np.random.randint(-self.range_len, self.range_len)
                # add the obverse segment and random contents to indices
                self.indices.append((prev_index, i, j))
            torch.save(self.indices, self.indices_cache)

    def __len__(self):
        # one true pair and one false pair
        return len(self.indices) * 2

    def __getitem__(self, index):
        index = index // 2

        prev_index, i, j = self.indices[index]

        seg = self.seg_tokens[i]
        
        # select some contents from previous segmentation
        if prev_index < 0:
            prev_seg = self.seg_tokens[i-1]
            seg = prev_seg[prev_index:] + seg
        else:
            seg = seg[prev_index:]
        if len(seg) < self.seq_len:
            new_len = self.seq_len - len(seg)
            seg = seg + self.seg_tokens[i+1][:new_len]
        
        label = index % 2
        clone = self.seg_tokens[i] if label == 1 else self.seg_tokens[j]

        seg = seg[:self.seq_len]
        clone = clone[:self.seq_len]

        x = torch.tensor([self.tokenizer.bos_token_id] + seg \
            + [self.tokenizer.bos_token_id] + clone, dtype=torch.long)
        label = torch.tensor([label]).type(torch.long)
        return x, label

