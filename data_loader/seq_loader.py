import glob
import hashlib
from os.path import join

import torch
from torch.utils.data import Dataset

from data_loader.setup_BPE import get_tokenizer


class SeqDataset(Dataset):
    """
    Load tokens sequentially   
    """

    def __init__(self, token_path, seq_len, max_len=100000):
        tk_files = glob.glob(join(token_path, '*.tk'))
        assert len(tk_files) > 0, 'No token files found'
        tk_files.sort()
        
        self.data = {}
        self.selected_files = []
        self.seq_len = seq_len

        # apply bpe
        tokenizer = get_tokenizer()
        # load all files
        for f in tk_files:
            with open(f, 'r', encoding='utf-8') as content:
                tokens = tokenizer.tokenize(content.read())
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                if len(token_ids) < seq_len + 1:
                    continue
                token_ids = SeqDataset._prune_tokens(
                    token_ids, tokenizer.sep_token_id)
                key = hashlib.md5(f.encode()).hexdigest()
                self.data[key] = token_ids
                # map key to index
                self.selected_files.append(key)
                for i in range(1, len(tokens) - seq_len):
                    self.selected_files.append(i)
                if len(self.selected_files) > max_len:
                    break

    def __len__(self):
        return len(self.selected_files)

    def __getitem__(self, index):
        hash_or_i = self.selected_files[index]
        if isinstance(hash_or_i, int):
            hash = self.selected_files[index - hash_or_i]
            i = hash_or_i
        else:
            hash = hash_or_i
            i = 0
        tokens = self.data[hash]
        x = torch.tensor(tokens[i:i+self.seq_len], dtype=torch.long)
        y = torch.tensor(tokens[i+1:i+1+self.seq_len], dtype=torch.long)
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
