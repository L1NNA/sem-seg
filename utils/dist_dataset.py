import glob
from os.path import join, basename, exists

import numpy as np
import torch
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import Dataset

from utils.config import Config


class DistDataset(Dataset):
    """
    The dataset that supports DDP
    """

    def __init__(self, config:Config, name, stage):
        self.config = config
        self.name = name
        self.stage = stage
        if config.model == 'autobert' \
          and config.bert_name is not None \
          and config.bert_name != 'graphcodebert':
            self.name += '_' + config.bert_name
        self.name += '_' + stage

    def pipeline(self):
        if self.config.is_host:
            self.load_data()
        if self.config.distributed:
            dist.barrier()
        self.load_cache()
        if self.config.distributed:
            dist.barrier()

    def load_data(self):
        raise Exception('Not implemented')

    def load_cache(self):
        raise Exception('Not implemented')
