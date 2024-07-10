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

def add_args(args):
    args.add_argument('--skip_label', type=int, default=0,
                        help='number of labels to skip')
    args.add_argument('--skip_seg', action='store_true',
                        help='if to skip segmented windows')


class LabelingDataset(DistDataset):
    """
    Classifies the type of the current window
    """

    def __init__(self, config:Config, stage):
        if config.n_windows == 0:
            config.n_windows = 1
        super().__init__(config, f'{config.data}_{config.seq_len}', stage)
        
        self.n_windows = config.n_windows
        self.window_size = self.seq_len // config.n_windows
        self.skip_label = config.skip_label
        self.skip_seg = config.skip_seg

    def __getitem__(self, index):
        i, j = self.indices[index]

        tokens = []
        labels = []
        for token, _, label in self.get_all(i, j, self.seq_len):
            tokens.append(token)
            labels.append(label)
        label = label_seg(labels)
        x = torch.tensor(tokens, dtype=torch.long)
        y = torch.tensor([label], dtype=torch.long)
        return x, y

    def load_data(self):
        if exists(self.indices_cache_path):
            return

        print('Loading dataset...')
        for i, seg_file in enumerate(self.tokens):
            seg_file:List[Tuple[List[str], str]]
            token_ids = [token for tokens,_ in seg_file for token in tokens]
            if len(token_ids) < self.seq_len:
                continue

            for j in range(0, len(token_ids)-self.seq_len, self.seq_len):
                self.indices.append((i, j))
                

            # j = 0 # index of the first token of current seg
            # last = 0 # index of last cached token 
            # total = sum(len(tokens) for tokens,_ in seg_file)
            # for tokens, label in seg_file:
            #     label_type = get_seg_type(label).value
            #     # check for skip label
            #     if label_type < self.skip_label:
            #         j += len(tokens) # jump to next seg
            #         last = j-(j % self.window_size) # jump to next sliding window
            #         continue
            #     # cache from last cached token
            #     # if skip seged window, jump to next window
            #     start = last + self.window_size if self.skip_seg and last < j else last
            #     # cache until the last token of the current seg
            #     last = j + len(tokens) - self.seq_len
            #     # if no need to skip seged window, cache until next window
            #     if not self.skip_seg and (j +len(tokens) + self.window_size) < total:
            #         last += self.window_size
            #     for k in range(start, last, self.seq_len):
            #         self.indices.append((i, j, label_type))
            #     last = k + self.seq_len
            #     j += len(tokens) # jump to next seg
            #     if last < j: # jump to last sliding window of current seg
            #         last = j-(j % self.window_size) 
        
        torch.save(self.indices, self.indices_cache_path)