import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
import torch.distributed as dist

from utils.config import Config
from utils.setup_BPE import get_tokenizer
from utils.dist_dataset import DistDataset
from utils.distributed import distribute_dataset, gather_tensors


class CloneSearch:

    def __init__(self, config:Config):
        self.config = config
        self.sources = None

    def load_sources(self, dist_dataset, model:DDP):
        _clone_dataset = _CloneDataset(self.config, dist_dataset)
        _clone_loader = distribute_dataset(self.config, _clone_dataset, self.config.batch_size)
        sources = None

        iterator = tqdm(_clone_loader, desc='Cache clones') \
            if self.config.is_host else _clone_loader
        for x, masking in iterator:
            x = x.to(model.device)
            masking = masking.to(model.device)

            y = model.module.representation(x, masking) if isinstance(model, DDP) else model.representation(x, masking)
            y = y[:, 0, :]
            y = y / y.norm(dim=1, keepdim=True)
            y = y
            if sources is None:
                sources = y
            else:
                sources = torch.concat((sources, y), dim=0)

        self.sources = gather_tensors(sources, self.config)

    def clone_search(self, x):
        y = self.sources.to(x.device)
        x = x / x.norm(dim=1, keepdim=True)
        return torch.matmul(x, y.transpose(0, 1))


class _CloneDataset(Dataset):

    def __init__(self, config:Config, dist_dataset:DistDataset):
        
        seq_len = config.seq_len
        self.pad_token_id = get_tokenizer().pad_token_id
        self.sources = []
        for tokens, label in dist_dataset.get_sources():
            if len(tokens) > seq_len:
                tokens = tokens[:seq_len]
            elif len(tokens) < seq_len:
                tokens = tokens + [self.pad_token_id] * (seq_len - len(tokens))
            
            self.sources.append(tokens)

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, index):
        x = torch.tensor(self.sources[index])
        masking = (x != self.pad_token_id).long()
        return x, masking