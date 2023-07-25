import glob
import hashlib
from os.path import join

import redis
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.config import Config

from data_loader.setup_BPE import get_tokenizer


FILENAMES = 'filenames'


class SeqRedisDataset(Dataset):
    """
    Load tokens sequentially from redis
    """

    def __init__(self, config:Config, max_len=100000):
        self.config = config
        self.r = redis.Redis()
        self.caching = []
        self.selected_files = {}
        self.max_len = max_len

    def load_data(self):
        num_files = self.r.lrange(FILENAMES, 0, -1)
        seq_len = self.config.seq_len
        
        # load all files
        for f in tqdm(num_files, desc='Loading data'):
            tokens = self.r.lrange(f, 0, -1)
            if len(tokens) < seq_len + 1:
                continue
            
            tokens = [int(v) for v in tokens]
            self.selected_files[f] = tokens
            # map key to index
            self.caching.append(f)
            for i in range(1, len(tokens)-seq_len):
                self.caching.append(i)
            if len(self.caching) > self.max_len:
                break

    def __len__(self):
        return len(self.caching)

    def __getitem__(self, index):
        seq_len = self.config.seq_len
        f_or_i = self.caching[index]
        i = f_or_i if isinstance(f_or_i, int) else 0
        f = self.caching[index - i]

        tokens = self.selected_files[f]
        x = torch.tensor(tokens[i:seq_len+i], dtype=torch.long)
        y = torch.tensor(tokens[i+1:seq_len+i+1], dtype=torch.long)
        return x, y

    @staticmethod
    def _prune_tokens(tokens, seq_token_id):
        result = []
        prev = None
        for tk in tokens:
            if tk == seq_token_id and \
                (prev is None or prev == tk):
                continue
            result.append(tk)
            prev = tk
        if result[-1] != seq_token_id:
            result.append(seq_token_id)
        return result

    @staticmethod
    def _cache_tokens(f):
        r = redis.Redis()
        with open(f, 'r', encoding='utf-8') as content:
            tokens = tokenizer.tokenize(content.read())
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            token_ids = _prune_tokens(
                token_ids, tokenizer.sep_token_id)
            key = basename(f)
            r.rpush(key, *token_ids)
            r.rpush(FILENAMES, key)
        r.close()
