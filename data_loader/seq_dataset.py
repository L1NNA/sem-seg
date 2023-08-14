import glob
from os.path import join, basename

import torch
from torch.utils.data import Dataset

from data_loader.setup_BPE import get_tokenizer
from utils.config import Config


class SeqDataset(Dataset):
    """
    Load tokens sequentially   
    """

    def __init__(self, config:Config, name='train', max_len=100000):
        self.config = config
        self.token_path = join(config.data_path, name)
        self.seq_len = config.seq_len
        self.max_len = max_len
        self.seq_token_id = get_tokenizer().sep_token_id
        
        self.segments = []
        self.tokens = {}
        self.labels = {}

    def load_data(self):
        x_files = glob.glob(join(self.token_path, '*.x.pt'))
        y_files = glob.glob(join(self.token_path, '*.y.pt'))
        y_map = {basename(f)[:-5]:f for f in y_files}

        for x_file in x_files:
            name = basename(x_file)[:-5]
            y_file = y_map[name]

            x = torch.load(x_file)
            y = torch.load(y_file)

            if len(x) < self.seq_len + 1:
                continue

            self.tokens[name] = x
            self.labels[name] = y

            self.segments.append(name)
            self.segments.extend(range(1, len(x)-self.seq_len-1))
            if len(self.segments) > self.max_len:
                break

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        key_or_int = self.segments[index]
        i = key_or_int if isinstance(key_or_int, int) else 0
        key = self.segments[index - i]

        tokens = self.tokens[key]
        labels = self.labels[key][i:i+self.seq_len]
        labels = torch.tensor(labels).bool()

        x = torch.tensor(tokens[i:i+self.seq_len], dtype=torch.long)
        y = torch.tensor(tokens[i+1:i+1+self.seq_len], dtype=torch.long)
        y = torch.masked_fill(y, labels, self.seq_token_id)
        
        return x, y

    
