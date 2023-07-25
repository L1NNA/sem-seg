import glob
import hashlib
from os.path import join

import torch
import numpy as np
from torch.utils.data import Dataset

from data_loader.setup_BPE import get_tokenizer


class BinaryDataset(Dataset):
    """
    Classifies whether the current window has
    a segment boundary or not 
    """

    def __init__(self, token_path, seq_len, max_len=100000):
        tk_files = glob.glob(join(token_path, '*.tk'))
        assert len(tk_files) > 0, 'No token files found'
        tk_files.sort()
        
        self.data = {}
        self.segments = []
        self.seq_len = seq_len

        # apply bpe
        tokenizer = get_tokenizer()
        self.seq_token_id = tokenizer.sep_token_id
        # load all files
        for f in tk_files:
            with open(f, 'r', encoding='utf-8') as content:
                tokens = tokenizer.tokenize(content.read())
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                token_ids = BinaryDataset._prune_tokens(
                    token_ids, tokenizer.sep_token_id) 
                if len(token_ids) < seq_len + 1:
                    continue
                
                # find sep tokens
                indices = []
                segments = []
                for i in range(0, len(token_ids)-seq_len, seq_len):
                    if token_ids[i:i+seq_len] \
                     .find(tokenizer.sep_token_id) != -1:
                        segments.append(i)
                    else:
                        indices.append(i)
                # sample n elements from indices
                # TODO make sure it's the same acorss all processes
                indices = np.random.choice(indices, len(segments), replace=False)
                segments.extend(indices)
                # add segments to data
                for i in segments:
                    key = hashlib.md5(f'{f}-{i}'.encode()).hexdigest()
                    self.data[key] = token_ids[i:i+seq_len]
                    self.segments.append(key)
                
                if len(self.segments) > max_len:
                    break

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        hash = self.segments[index]
        tokens = self.data[hash]
        is_sep = tokens.find(self.sep_token_id) != -1
        x = torch.tensor(tokens, dtype=torch.long)
        y = torch.tensor([is_sep]).type(torch.int)
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
