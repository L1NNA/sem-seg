import glob
from os.path import join, exists

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

from utils.config import Config
from utils.setup_BPE import get_tokenizer


def load_classification(files_path, range_len=100):
    seg_files = glob.glob(join(files_path, '*.pt'))

    training = []
    segments = []

    for seg_file in tqdm(seg_files, desc='Load files'):
        segs = torch.load(seg_file)

        # randomly pick a number from 1 to len(segs) - 1
        if len(segs) <= 2:
            continue
        index = np.random.randint(1, len(segs) - 1)
        seg_ori, label = segs[index]
        prev_seg, prev_label = segs[index - 1]
        next_seg, next_label = segs[index + 1]

        seg = seg_ori
        
        # select some contents from previous segmentation
        prev_index = np.random.randint(-range_len, range_len)
        if prev_index < 0:
            seg = prev_seg[prev_index:] + seg
        else:
            seg = seg[prev_index:]

        # select some contents from next segmentation
        post_index = np.random.randint(-range_len, range_len)
        if post_index < 0:
            seg = seg[:post_index]
        else:
            seg += next_seg[:post_index]

        # add the altered segmentation to training cache
        training.append((seg, label))
        # add the original segments to labels cache
        segments.append((seg_ori, label))
        segments.append((prev_seg, prev_label))
        segments.append((next_seg, next_label))

    return training, segments


def get_random_index(span, indices):
    possible_indices = np.setdiff1d(np.arange(span), indices, assume_unique=False)
    return np.random.choice(possible_indices)


class ClassificationDataset(Dataset):

    def __init__(self, files_path, name, range_len=100, limit=8000):
        self.files_path = join(files_path, name)
        self.token_cache = join(files_path, 'cache',
            f'{name}_classification_tokens.pt')
        self.indices_cache = join(files_path, 'cache',
            f'{name}_classification_{range_len}_indices.pt')
        self.range_len = range_len
        self.seg_tokens = []
        self.indices = []
        self.limit = limit
        self.name = name

    def load_data(self):
        if exists(self.token_cache):
            self.seg_tokens = torch.load(self.token_cache)
            print(self.name, 'tokens loaded')
        else:
            tokenizer = get_tokenizer()
            seg_files = glob.glob(join(self.files_path, '*.pt'))

            # load files and tokenization
            for seg_file in tqdm(seg_files, desc='Load files'):
                segs = torch.load(seg_file)
                for seg, _ in segs:
                    tokens = tokenizer.encode(seg, add_special_tokens=False)
                    if len(tokens) < 1000 or len(tokens) > 10000:
                        continue
                    self.seg_tokens.append(tokens)
            print('Size: ', len(self.seg_tokens))
            torch.save(self.seg_tokens, self.token_cache)

        if exists(self.indices_cache):
            self.indices = torch.load(self.indices_cache)
            print(self.name, 'indices loaded')
        else:
            span = len(self.seg_tokens)
            if self.limit < span-2:
                indices = np.random.choice(range(1, span-1), self.limit, replace=False)
            else:
                indices = [*range(1, span-1)]
            for i in tqdm(indices, desc='Load indices'):
                # select another segmentation, such that (i, j) has label 0
                j = get_random_index(span, [i-1, i, i+1])
                # select some tokens from previous segment
                prev_index = np.random.randint(-self.range_len, self.range_len)
                # select some tokens from next segment
                post_index = np.random.randint(-self.range_len, self.range_len)
                # add the obverse segment and random contents to indices
                self.indices.append((prev_index, post_index, i, j))
            torch.save(self.indices, self.indices_cache)

    def __len__(self):
        # one true pair and one false pair
        return len(self.indices) * 2

    def __getitem__(self, index):
        index = index // 2

        prev_index, post_index, i, j = self.indices[index]

        seg = self.seg_tokens[i]
        
        # select some contents from previous segmentation
        if prev_index < 0:
            prev_seg = self.seg_tokens[i-1]
            seg = prev_seg[prev_index:] + seg
        else:
            seg = seg[prev_index:]

        # select some contents from next segmentation
        if post_index < 0:
            seg = seg[:post_index]
        elif post_index > 0:
            next_seg = self.seg_tokens[i+1]
            seg += next_seg[:post_index]
        
        label = index % 2
        y = self.seg_tokens[i] if label == 1 else self.seg_tokens[j]
        x = torch.tensor(seg, dtype=torch.long)
        y = torch.tensor(y).type(torch.long)
        label = torch.tensor([label]).type(torch.long)
        return x, y, label

