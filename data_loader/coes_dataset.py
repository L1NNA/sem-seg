import glob
from os.path import join, exists

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

from utils.config import Config
from utils.setup_BPE import get_tokenizer
from utils.dist_dataset import DistDataset
from utils.label_utils import get_seg_type, get_most_common_label


class CoEsDataset(DistDataset):
    """
    Chain-of-experts dataset
    """

    def __init__(self, config:Config, stage):
        super().__init__(config, f'{config.data}_{config.seq_len}_{config.n_windows}', stage)
        self.n_windows = config.n_windows
        self.window_len = self.seq_len // self.n_windows
        self.pad_token_id = get_tokenizer().pad_token_id
        self.label_reverse_map = {}

    def _to_clone_index(self, labels):
        return [self.label_reverse_map[label] if label in self.label_reverse_map else -1 for label in labels]

    def __getitem__(self, index):
        i, j = self.indices[index]
        tokens = []
        segs = []
        labels = []
        for token, seg, label in self.get_all(i, j, self.seq_len):
            tokens.append(token)
            segs.append(seg)
            labels.append(label)

        x = torch.tensor(tokens, dtype=torch.long)
        seg_tensor = torch.tensor(segs, dtype=torch.long).reshape(self.n_windows, -1)
        seg_tensor = seg_tensor.max(dim=1).values

        sw_labels = [get_most_common_label(labels[self.window_len*w:self.window_len*(w+1)]) for w in range(self.n_windows)]
        label_tensor = torch.tensor([get_seg_type(label).value for label in sw_labels], dtype=torch.long) # w

        if self.stage == 'train':
            masking = torch.tensor(self._to_clone_index(labels)).reshape(self.n_windows, -1) # w x win_len
            clone_tensor = torch.tensor(self._to_clone_index(sw_labels), dtype=torch.long) # w
            masking = masking == clone_tensor.unsqueeze(-1).repeat(1, self.window_len)  # w x win_len
            masking = masking.long().reshape(-1) # seq_len
            return x, masking, seg_tensor, label_tensor
        else:
            clone_tensor = torch.tensor(self._to_clone_index(sw_labels), dtype=torch.long)
            return x, seg_tensor, label_tensor, clone_tensor

    def pipeline(self):
        super().pipeline()

        # reverse map label to its index
        for index, (_,label) in enumerate(self.get_sources()):
            self.label_reverse_map[label] = index

    def load_data(self):
        
        if exists(self.indices_cache_path):
            return
        
        print('Loading dataset...')
        for i, seg_file in enumerate(self.tokens):

            seg_file:List[Tuple[List[str], str]]
            labels = [label for tokens,label in seg_file for _ in tokens]
            if len(labels) < self.seq_len:
                continue

            for j in range(0, len(labels)-self.seq_len, self.window_len):
                lable = get_most_common_label
                self.indices.append((i, j))

        torch.save(self.indices, self.indices_cache_path)
