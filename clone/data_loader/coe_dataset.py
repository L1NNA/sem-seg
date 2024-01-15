import glob
from os.path import join, exists

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

from utils.config import Config
from utils.setup_BPE import get_tokenizer
from utils.dist_dataset import DistDataset
from labeling.data_loader.label_utils import get_seg_type


class COEDataset(DistDataset):
    """
    Chain-of-experts dataset
    """

    def __init__(self, config:Config, stage):
        super().__init__(config, f'{config.data}_{config.seq_len}_{config.n_windows}', stage)
        self.config = config
        self.files_path = join(config.data_path, stage)
        self.seq_len = config.seq_len
        self.n_windows = config.n_windows
        self.max_len = self.seq_len * self.n_windows
        self.pad_token_id = get_tokenizer().pad_token_id
        self.cache_path = join(config.data_path, 'cache',  config.data, f'{self.name}.pt')
        self.cache = {}

    def __len__(self):
        return len(self.cache['indices'])

    def __getitem__(self, index):
        i, j = self.cache['indices'][index]
        tokens, segs, labels = self.cache['segments'][i]
        x = torch.tensor(tokens[j:j+self.max_len], dtype=torch.long)
        y = torch.tensor(tokens[j:j+self.max_len], dtype=torch.long)
        seg_tensor = torch.tensor(segs, dtype=torch.long).reshape(self.n_windows, -1)
        seg_tensor = seg_tensor.max(dim=1).values
        label_ori = torch.tensor(labels, dtype=torch.long)
        label_tensor = label_ori.reshape(self.n_windows, -1)
        label_tensor = label_tensor.max(dim=1).values
        
        masking = label_tensor.unsqueeze(-1).repeat(1, self.seq_len).reshape(-1) == label_ori
        y = torch.where(masking, y, torch.full((self.max_len, ), self.pad_token_id, dtype=torch.long))
        y_mask = masking.long()
        return x, y, y_mask, seg_tensor, label_tensor

    def load_data(self):
        
        if exists(self.cache_path):
            return
        
        print('Loading dataset')
        self.cache['indices'] = []
        self.cache['segments'] = []
        for seg_file in tqdm(glob.glob(join(self.files_path, '*.pt'))):

            token_ids, seg_ids, label_ids = COEDataset.build_file(seg_file)

            if len(token_ids) < self.max_len:
                continue
            
            i = len(self.cache['segments'])
            self.cache['segments'].append((token_ids, seg_ids, label_ids))
            for j in range(0, len(token_ids)-self.max_len, self.seq_len):
                self.cache['indices'].append((i, j))

        torch.save(self.cache, self.cache_path)
        indices = len(self.cache['indices'])
        print(f'Total number of {self.stage} samples: {indices}')

    @staticmethod
    def build_file(seg_file):
        tokenizer = get_tokenizer()
        segs = torch.load(seg_file)
        token_ids = []
        seg_ids = []
        label_ids = []
        for seg, label in segs:
            tokens = tokenizer.encode(seg, add_special_tokens=False)
            token_ids.extend(tokens)
            seg_ids.extend([0] * (len(tokens) - 1) + [1])
            label_type = get_seg_type(label)
            label_ids.extend([label_type.value] * len(tokens))
        return token_ids, seg_ids, label_ids

    def load_cache(self):
        if len(self.cache) > 0:
            return
        assert exists(self.cache_path), 'cache not found'
        self.cache = torch.load(self.cache_path)
        if self.config.is_host:
            print(f'Total number of {self.stage} samples: {len(self.cache)}')