from os.path import exists

import torch

from utils.config import Config
from utils.setup_BPE import get_tokenizer
from data_loader.dist_dataset import DistDataset, get_simple_label, get_seg_type


class COEDataset(DistDataset):
    """
    Chain-of-experts dataset
    """

    def __init__(self, config:Config, stage):
        super().__init__(config, f'{config.seq_len}_{config.n_windows}', stage)
        self.n_windows = config.n_windows
        self.window_len = self.seq_len // self.n_windows
        self.pad_token_id = get_tokenizer().pad_token_id

    def __getitem__(self, index):
        i, j = self.indices[index]
        tokens = []
        segs = []
        labels = []
        src_tokens = []
        for token, seg, label_type, src_token in self.get_all(i, j, self.seq_len, self.pad_token_id):
            tokens.append(token)
            segs.append(seg)
            labels.append(label_type)
            src_tokens.append(src_token)

        x = torch.tensor(tokens, dtype=torch.long)
        x_mask = torch.ones_like(x, dtype=torch.long)
        y = torch.tensor(src_tokens, dtype=torch.long)
        y_mask = torch.where(y == self.pad_token_id, 0, 1)
        seg_tensor = torch.tensor(segs, dtype=torch.long).reshape(self.n_windows, -1)
        seg_tensor = seg_tensor.max(dim=1).values
        label_tensor = torch.tensor(labels, dtype=torch.long).reshape(self.n_windows, -1)
        label_tensor = label_tensor.max(dim=1).values
        y_mask = torch.where(y == self.pad_token_id, 0, 1)

        return x, x_mask, seg_tensor, label_tensor, y, y_mask

    def load_data(self):
        
        if exists(self.indices_cache_path):
            return
        
        print('Loading dataset...')
        for i, seg_file in enumerate(self.tokens):

            seg_file:List[Tuple[List[str], str, List[str]]]
            get_labels = lambda tokens:[0] * (len(tokens)-1) + [1]
            all_labels:List[int] = [label for tokens,_,_ in seg_file for label in get_labels(tokens)]
            if len(all_labels) < self.seq_len:
                continue

            for j in range(0, len(all_labels)-self.seq_len, self.window_len):
                if self.stage == 'train': # skip inputs with no boundaries for training
                    label = 1 if 1 in all_labels[j:j+self.seq_len] else 0
                    if label == 0:
                        continue
                self.indices.append((i, j))

        torch.save(self.indices, self.indices_cache_path)
