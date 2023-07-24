
from .seq_loader import SeqDataset
from utils.config import Config
from .setup_BPE import get_tokenizer


def load_dataset(config:Config):

    if config.data == "seq":
        return SeqDataset(config.data_path, config.seq_len)
    return None

def load_tokenizer(config:Config):
    tokenizer = get_tokenizer()
    config.vocab_size = tokenizer.vocab_size