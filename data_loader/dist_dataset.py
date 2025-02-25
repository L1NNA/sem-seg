import glob, re
from os.path import join, basename, exists
from typing import List, Tuple, Iterator
from enum import Enum

import numpy as np
import torch
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import Dataset

from utils.config import Config
from utils.setup_BPE import get_tokenizer


class SegType(Enum):
    BOOTSTRAP = 0
    LOCAL = 1
    THIRD_PARTY = 2

BOOTSTRAPS = (
    'webpack/universalModuleDefinition',
    'webpack/bootstrap',
    'webpack/startup',
    '/runtime/[a-zA-Z]+',
    '/external',
)
NODE_MODULES = '/node_modules/'
UNK = '<UNK>'

def get_simple_label(label:str) -> int:
    
    if label is None:
        return None
    for boostrap in BOOTSTRAPS:
        if re.match('^webpack://.*' + boostrap + '.*', label):
            return SegType.BOOTSTRAP.value
    return SegType.LOCAL.value

def get_seg_type(label:str) -> int:
    """
    Map the source name to a segmentation type
    """
    if label is None:
        return None

    # The segment refers to a node module package
    if label.find(NODE_MODULES) > -1:
        return SegType.THIRD_PARTY.value

    for boostrap in BOOTSTRAPS:
        if re.match('^webpack://.*' + boostrap + '$', label):
            return SegType.BOOTSTRAP.value

    if label == UNK:
        return None
    
    # Local files
    if label.find('./') > -1:
        return SegType.LOCAL.value

    return None


class DistDataset(Dataset):
    """
    The dataset that supports DDP and tokenization
    """

    def __init__(self, config:Config, name, stage):
        self.config = config
        self.name = name
        self.stage = stage
        self.seq_len = config.seq_len

        tokens_path_name = stage
        if config.bert_name is not None \
          and config.bert_name != 'graphcodebert':
            self.name += '_' + config.bert_name
            tokens_path_name += '_' + config.bert_name
        self.name += '_' + stage

        self.tokens_path = join(config.data_path, 'tokens', f'{tokens_path_name}.pt')
        self.tokens:List[List[Tuple[List[str], str, List[str]]]] = []
        self.files_path = join(config.data_path, stage)

        self.indices_cache_path = join(config.data_path, 'cache', f'{self.name}.indices.pt')
        self.indices:List = []

    def pipeline(self):
        # tokens
        if self.config.is_host:
            self.cache_tokens()
        if self.config.distributed:
            dist.barrier()
        self.load_tokens()
        if self.config.distributed:
            dist.barrier()

        # indices
        if self.config.is_host:
            self.load_data()
        if self.config.distributed:
            dist.barrier()
        self.load_cache()
        if self.config.distributed:
            dist.barrier()
        if self.config.is_host:
            print(f'Total number of {self.stage} samples: {len(self)}')

    def cache_tokens(self):
        if exists(self.tokens_path):
            return
        
        tokenizer = get_tokenizer()
        iterator = tqdm(glob.glob(join(self.files_path, '*.pt')), desc='Loading tokens')
        for seg_file in iterator:
            segs = torch.load(seg_file)
            seg_tokens:List[Tuple[List[str], str, List[str]]] = []
            for seg, label, src in segs:
                label_type = get_seg_type(label)
                if label_type is None: # skip unknown sources
                    continue
                if len(src) >= len(seg) * 1.5:
                    src = src[:int(len(seg) * 1.5)]
                tokens:List[str] = tokenizer.encode(seg, add_special_tokens=False)
                src_tokens:List[str] = tokenizer.encode(src, add_special_tokens=False)
                seg_tokens.append((tokens, label_type, src_tokens))
            self.tokens.append(seg_tokens)
        torch.save(self.tokens, self.tokens_path)

    def load_tokens(self):
        if len(self.tokens) > 0:
            return
        assert exists(self.tokens_path), 'tokens cache not found'
        self.tokens = torch.load(self.tokens_path)

    def load_data(self):
        raise Exception('Not implemented')

    def load_cache(self):
        if len(self.indices) > 0:
            return
        assert exists(self.indices_cache_path), 'indices cache not found'
        self.indices = torch.load(self.indices_cache_path)

    def __len__(self):
        return len(self.indices)

    def get_all(self, i, j, seq_len, pad_token_id=-1) -> Iterator[Tuple[int, int, str, int]]:
        seg_file = self.tokens[i]
        start, end = 0, j+seq_len

        for tokens, label, src_tokens in seg_file:
            for k, token in enumerate(tokens):
                if start < j:
                    start += 1
                    continue
                if start < end:
                    seg = 0 if k < len(tokens)-1 else 1
                    src_token = src_tokens[k] if k < len(src_tokens) else pad_token_id
                    yield token, seg, label, src_token
                    start += 1
                else:
                    break
    
    def get_sources(self):
        for seg_file in self.tokens:
            for tokens, label, src_tokens in seg_file:
                label_type = get_simple_label(label)
                # if label_type.value >= self.config.skip_label:
                yield tokens, label, src_tokens



