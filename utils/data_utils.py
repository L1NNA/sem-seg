import torch.distributed as dist

from data_loader.coe_dataset import COEDataset
from utils.setup_BPE import get_tokenizer
from data_loader.dist_dataset import DistDataset, SegType
from utils.config import Config


def get_output_dim(config:Config):
    if config.do_seg and config.do_cls and config.do_ccr:
        return (2, len(SegType), -1)
    elif config.do_seg and config.do_cls:
        return (2, len(SegType), -1)
    elif config.do_seg:
        return 2
    elif config.do_cls:
        return len(SegType)
    elif config.do_ccr:
        return -1

def load_dataset(config:Config):
    train, valid, test = _load_all(COEDataset, config)
    _load_cache(train, config)
    _load_cache(valid, config)
    _load_cache(test, config)
    return train, valid, test, get_output_dim(config)

def _load_all(clazz, config:Config):
    train, valid, test = None, None, None
    if config.training:
        train = clazz(config, 'train')
        valid = clazz(config, 'valid')
    if config.validation and valid is None:
        valid = clazz(config, 'valid')
    if config.testing:
        test = clazz(config, 'test')
    return train, valid, test

def _load_cache(dataset:DistDataset, config:Config):
    if dataset is None:
        return
    dataset.pipeline()

def load_tokenizer(config:Config):
    if config.bert_name is None \
      and (config.model == 'longformer' \
      or config.model == 'sentbert'):
        config.bert_name = config.model
    tokenizer = get_tokenizer(config)
    config.vocab_size = tokenizer.vocab_size