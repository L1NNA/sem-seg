from os.path import join
from typing import List, Iterator

import torch

from utils.config import Config
from utils.checkpoint import load_checkpoint
from data_loader.setup_BPE import get_tokenizer

from tqdm import tqdm


class Tokenizer:

    def __init__(self, config:Config, model):
        self.config:Config = config
        checkpoint = join(config.checkpoint, config.model_name + '.pt')
        load_checkpoint(checkpoint, self.model, None, None)
        self.tokens:List[str] = None
        self.segment_labels:List[int] = None
        self.lablels:List[str] = None

    def encode(self, text):
        tokenizer = get_tokenizer()
        self.tokens = tokenizer.encode(text)
        assert len(self.tokens) > self.config.seq_len, "Input is too short"

    def load_file(self, seg_file):
        self.tokens = []
        self.lablels = []
        tokenizer = get_tokenizer()
        segs = torch.load(seg_file)
        index = 0
        for seg, _ in segs:
            tokens = tokenizer.encode(seg, add_special_tokens=False)
            self.tokens.extend(tokens)
            i += len(tokens) - 1
            self.lablels.append(i)

    def greedy_segmentation(self):
        assert self.tokens is not None, "Please run encode first"
        segment_labels = []
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
                segment_labels.append(i)
                # if there is a segment boundary, skip the whole window
                skip = i + self.config.seq_len
        segment_labels.append(None)
        return segment_labels

    def full_segmentation(self):
        assert self.tokens is not None, "Please run encode first"
        segment_labels = [0] * self.config.seq_len
        for i in tqdm(range(len(self.tokens)-self.config.seq_len), 'Segmenting'):
            tokens = self.tokens[i:i+self.config.seq_len]
            input_ids = torch.tensor(tokens).unsqueeze(0)
            with torch.no_grad():
                output = self.model(input_ids)
            output = torch.argmax(torch.softmax(output.squeeze(0))).item()
            if output == 1:
                for j in range(-1, -1-self.config.seq_len, -1):
                    segment_labels[j] += 1
        return segment_labels
        
    def get_segments(self) -> Iterator[str]:
        if self.segment_labels is None:
            self.segmentation()
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

