import torch.distributed as dist

from utils.config import Config

from .binary_dataset import BinaryDataset
from .seq_dataset import SeqDataset
from .setup_BPE import get_tokenizer


DATASET_MAP = {
    'seq': (SeqDataset, lambda config: config.vocab_size),
    'binary': (BinaryDataset, lambda _: 2),
}


def _load_all(clazz, config:Config):
    train, valid, test = None, None, None
    if config.training:
        train = clazz(config, 'train')
        valid = clazz(config, 'valid')
    if config.testing:
        test = clazz(config, 'test')
    return train, valid, test

def _load_cache(dataset, config:Config):
    if dataset is None:
        return
    dataset.load_data()
    if config.distributed:
        dist.barrier()
    # check if dataset has load_cache method
    if hasattr(dataset, 'load_cache'):
        dataset.load_cache()
    if config.distributed:
        dist.barrier()

def load_dataset(config:Config):
    clazz, get_output_dim = DATASET_MAP[config.data]
    train, valid, test = _load_all(clazz, config)
    _load_cache(train, config)
    _load_cache(valid, config)
    _load_cache(test, config)
    return train, valid, test, get_output_dim(config)

def load_tokenizer(config:Config):
    tokenizer = get_tokenizer()
    config.vocab_size = tokenizer.vocab_size