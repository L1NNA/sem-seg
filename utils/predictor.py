from data_loader.tokenizer import Tokenizer
from utils.config import Config


def segmentation(config:Config, model):

    with open(config.segmentation, 'r') as f:
        code = f.read()

    tokenizer = Tokenizer(config, model)
    tokenizer.encode(code)
    for segment in tokenizer.get_segments():
        print(segment)
        print('----------------------------------')
