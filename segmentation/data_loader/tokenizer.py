from os.path import join
from typing import List, Iterator

import torch
from tqdm import tqdm

from utils.config import Config
from utils.checkpoint import load_checkpoint
from utils.setup_BPE import get_tokenizer


class Tokenizer:

    def __init__(self, config:Config, model):
        self.config:Config = config
        self.model = model
        self.tokens:List[int] = None
        self.segment_labels:List[int] = None
        self.offset_map = None
        self.lablels:List[str] = None

    def encode(self, text, labels):
        tokenizer = get_tokenizer()
        encoding = tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=False)
        self.tokens = encoding['input_ids']
        self.offset_map = encoding['offset_mapping']
        assert len(self.tokens) > self.config.seq_len, "Input is too short"
        self.labels = labels

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
            input_ids = input_ids.to(self.config.device)
            with torch.no_grad():
                output = self.model(input_ids)
            output = torch.argmax(torch.softmax(output.squeeze(0), dim=0), dim=0).item()
            if output == 1:
                segment_labels.append(i)
                # if there is a segment boundary, skip the whole window
                skip = i + self.config.seq_len
        segment_labels.append(None)
        return segment_labels

    def full_segmentation(self):
        assert self.tokens is not None, "Please run encode first"
        segment_labels = [0] * (self.config.seq_len - 1)
        for i in tqdm(range(len(self.tokens)-self.config.seq_len), 'Segmenting'):
            segment_labels.append(0)
            tokens = self.tokens[i:i+self.config.seq_len]
            input_ids = torch.tensor(tokens).unsqueeze(0)
            input_ids = input_ids.to(self.config.device)
            with torch.no_grad():
                output = self.model(input_ids)
            output = torch.argmax(torch.softmax(output.squeeze(0), dim=0), dim=0).item()
            if output == 1:
                for j in range(-1, -1-self.config.seq_len, -1):
                    segment_labels[j] += 1
        return segment_labels
        
    def get_segments(self, segment_labels) -> Iterator[str]:
        init = 0
        for index in segment_labels:
            tokens = self.tokens[init:index]
            yield get_tokenizer().decode(tokens)
            init = index
    
    def classfication(self, segment_labels, get_label):
        self.lablels = []
        for segment in self.get_segments(segment_labels):
            label = get_label(segment)
    
    def pipeline(self, text:str, method, get_label):
        self.encode(text)
        segments = self.greedy_segmentation()
        self.classfication()
        return self.lablels

