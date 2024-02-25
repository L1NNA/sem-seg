import glob
from os.path import join, exists

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

from utils.config import Config
from utils.setup_BPE import get_tokenizer
from utils.dist_dataset import DistDataset
from utils.label_utils import get_seg_type


class COEDataset(DistDataset):
    """
    Chain-of-experts dataset
    """

    def __init__(self, config:Config, stage):
        super().__init__(config, f'{config.data}_{config.seq_len}_{config.n_windows}', stage)
        self.n_windows = config.n_windows
        self.max_len = self.seq_len * self.n_windows
        self.pad_token_id = get_tokenizer().pad_token_id

    def __getitem__(self, index):
        i, j = self.indices[index]
        tokens = []
        segs = []
        labels = []
        for token, seg, label in self.get_all(i, j, self.max_len):
            tokens.append(token)
            segs.append(seg)
            labels.append(get_seg_type(label).value)

        x = torch.tensor(tokens, dtype=torch.long)
        y = torch.tensor(tokens, dtype=torch.long)
        seg_tensor = torch.tensor(segs, dtype=torch.long).reshape(self.n_windows, -1)
        seg_tensor = seg_tensor.max(dim=1).values
        label_ori = torch.tensor(labels, dtype=torch.long)
        label_tensor = label_ori.reshape(self.n_windows, -1)
        label_tensor = label_tensor.max(dim=1).values
        masking = label_tensor.unsqueeze(-1).repeat(1, self.seq_len).reshape(-1) == label_ori
        y = torch.where(masking, y, torch.full((self.max_len, ), self.pad_token_id, dtype=torch.long))
        y_mask = masking.long()

        del tokens, segs, labels

        return x, y, y_mask, seg_tensor, label_tensor

    def load_data(self):
        
        if exists(self.indices_cache_path):
            return
        
        print('Loading dataset...')
        for i, seg_file in enumerate(self.tokens):

            seg_file:List[Tuple[List[str], str]]
            token_ids = [token for tokens,_ in seg_file for token in tokens]
            if len(token_ids) < self.max_len:
                continue

            for j in range(0, len(token_ids)-self.max_len, self.seq_len):
                self.indices.append((i, j))

        torch.save(self.indices, self.indices_cache_path)
