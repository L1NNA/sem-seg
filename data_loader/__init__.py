import torch.distributed as dist

from utils.config import Config

from .binary_dataset import BinaryDataset
from .seq_dataset import SeqDataset
from .setup_BPE import get_tokenizer


DATASET_MAP = {
    'seq': SeqDataset,
    'binary': BinaryDataset
}


def _load_all(clazz, config:Config):
    train = clazz(config, 'train')
    valid = clazz(config, 'valid')
    test = clazz(config, 'test')
    return train, valid, test

def _load_cache(dataset):
    if dataset is None:
        return
    dataset.load_data()
    dist.barrier()
    # check if dataset has load_cache method
    if hasattr(dataset, 'load_cache'):
        dataset.load_cache()
    dist.barrier()

def load_dataset(config:Config):
    clazz = DATASET_MAP[config.data]
    train, valid, test = _load_all(clazz, config)
    _load_cache(train)
    _load_cache(valid)
    _load_cache(test)
    return train, valid, test

def load_tokenizer(config:Config):
    tokenizer = get_tokenizer()
    config.vocab_size = tokenizer.vocab_size