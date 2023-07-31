from os.path import join
from typing import List, Iterator

import torch

from utils.config import Config
from main import load_model
from utils.checkpoint import load_checkpoint
from data_loader.setup_BPE import get_tokenizer

from tqdm import tqdm


class Tokenizer:

    def __init__(self, config:Config):
        self.config:Config = config
        self.model = load_model(config)
        checkpoint = join(config.checkpoint, config.model_name + '.pt')
        load_checkpoint(checkpoint, self.model, None, None)
        self.tokens:List[str] = None
        self.segment_labels:List[int] = None
        self.lablels:List[str] = None

    def encode(self, text):
        tokenizer = get_tokenizer()
        self.tokens = tokenizer.encode(text)
        assert len(self.tokens) > self.config.seq_len, "Input is too short"

    def segmentation(self):
        assert self.tokens is not None, "Please run encode first"
        self.segment_labels = []
        skip = 0
        for i in tqdm(range(len(self.tokens)-self.config.seq_len), 'Segmenting'):
            if i < skip:
                continue
            tokens = self.tokens[i:i+self.config.seq_len]
            input_ids = torch.tensor(tokens).unsqueeze(0)
            with torch.no_grad():
                output = self.model(input_ids)
            output = torch.argmax(torch.softmax(output.squeeze(0))).item()
            if output == 1:
                self.segment_labels.append(i)
                # if there is a segment boundary, skip the whole window
                skip = i + self.config.seq_len
        self.segment_labels.append(None)

        
    def get_segments(self) -> Iterator[str]:
        assert self.segment_labels is not None, "Please run segmentation first"
        init = 0
        for index in self.segment_labels:
            tokens = self.tokens[init:index]
            yield get_tokenizer().decode(tokens)
            init = index
    
    def classfication(self):
        self.lablels = []
        for segment in self.get_segments():
            pass
        raise NotImplementedError
    
    def pipeline(self, text:str):
        self.encode(text)
        self.segmentation()
        self.classfication()
        return self.lablels

