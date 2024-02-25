from os.path import exists

import torch
import numpy as np

from utils.config import Config
from utils.dist_dataset import DistDataset


class SegmentationDataset(DistDataset):
    """
    Classifies whether the current window has
    a segment boundary or not.
    """

    def __init__(self, config:Config, stage):
        super().__init__(config, f'{config.data}_{config.seq_len}', stage)

    def load_data(self):
        
        if exists(self.indices_cache_path):
            return

        print('Loading dataset...')

        true_labels = 0
        for i, seg_file in enumerate(self.tokens):
            seg_file:List[Tuple[List[str], str]]
            get_labels = lambda tokens:[0] * (len(tokens)-1) + [1]
            all_labels:List[int] = [label for tokens,_ in seg_file for label in get_labels(tokens)]

            false_labels = [] # upsample
            for j in range(0, len(all_labels)-self.seq_len, self.seq_len):
                label = 1 if 1 in all_labels[j:j+self.seq_len] else 0
                if label == 1 and self.stage == 'train':
                    true_labels += 1 # upsample
                if label == 1 or self.stage != 'train':
                    self.indices.append((i, j, label))
                else:
                    false_labels.append((i, j, label)) # upsample

            if self.stage == 'train': # upsample
                if len(false_labels) > true_labels:
                    false_labels_indices = np.random.choice(len(false_labels), true_labels)
                    false_labels = [false_labels[i] for i in false_labels_indices]
                true_labels -= len(false_labels)
                self.indices.extend(false_labels)

        torch.save(self.indices, self.indices_cache_path)

    def __getitem__(self, index):
        i, j, label = self.indices[index]
        tokens = []
        for token,_,_ in self.get_all(i, j, self.seq_len):
            tokens.append(token)
        x = torch.tensor(tokens, dtype=torch.long)
        y = torch.tensor([label], dtype=torch.long)
        del tokens
        return x, y
