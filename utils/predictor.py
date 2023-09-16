from data_loader.tokenizer import Tokenizer
from utils.config import Config

import torch


def segmentation(config:Config, model):

    segs = torch.load(config.segmentation)
    code = ''
    labels = []
    for seg, _ in segs:
        code += seg
        labels.append(len(code))

    tokenizer = Tokenizer(config, model)
    cache = {}
    tokenizer.encode(code, labels)
    cache['mapping'] = tokenizer.offset_map
    cache['greedy_labels'] = tokenizer.greedy_segmentation()
    cache['full_labels'] = tokenizer.full_segmentation()
    cache['labels'] = labels

    torch.save(cache, './data2/cache/segmentation.pt')

    
