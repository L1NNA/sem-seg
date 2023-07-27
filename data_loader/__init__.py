from utils.config import Config

from .binary_dataset import BinaryDataset
from .seq_dataset import SeqDataset
from .setup_BPE import get_tokenizer


def load_dataset(config:Config):
    train, valid, test = None, None, None
    if config.data == "seq":
        train = SeqDataset(config, 'train')
        valid = SeqDataset(config, 'valid')
        test = SeqDataset(config, 'test')
    elif config.data == "binary":
        train = BinaryDataset(config, 'train')
        valid = BinaryDataset(config, 'valid')
        test = BinaryDataset(config, 'test')
    if train is not None:
        train.load_data()
    if valid is not None:
        valid.load_data()
    if test is not None:
        test.load_data()
    return train, valid, test

def load_tokenizer(config:Config):
    tokenizer = get_tokenizer()
    config.vocab_size = tokenizer.vocab_size