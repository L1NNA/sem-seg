"""
“Low-Resource” Text Classification: A Parameter-Free Classification Method with Compressors
https://aclanthology.org/2023.findings-acl.426
"""
import multiprocessing
from collections import defaultdict

import gzip
import numpy as np
from tqdm import tqdm_notebook

from .base_classification import BaseClassification
from utils.config import Config
from utils.setup_BPE import get_tokenizer


def compress_map(source, target, Cx1):
    Cx2 = len(gzip.compress(source.encode()))
    x1x2 = ' '.join([target, source])
    Cx1x2 = len(gzip.compress(x1x2.encode()))
    ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
    return ncd


class GzipClassification(BaseClassification):

    def classify(self, target:str):
        Cx1 = len(gzip.compress(target.encode()))

        with multiprocessing.Pool(50) as p:
            distance_from_x1 = p.starmap(compress_map, [(
                source, target, Cx1
            ) for source, _ in self.sources])
        sorted_idx = np.argsort(np.array(distance_from_x1))
        # very sparse, no need to take top k
        # top_k_class = self.sources[sorted_idx[:self.config.n_windows], 1]
        # predict_class = max(set(top_k_class), key=top_k_class.count)
        predict_class = self.sources[sorted_idx[0], 1]
        return predict_class
    

class CompressLoopClassification(BaseClassification):

    def __init__(self, config:Config):
        super().__init__(self, config)

    def get_sources(self, sources):
        for source in tqdm(sources, desc='Load sources'):
            for i in range(0, len(source[0])-self.config.seq_len+1, self.config.seq_len):
                self.sources.append([source[0][i:i+self.config.seq_len], source[1]])
        self.sources = np.array(self.sources)

    def classify(self, target: str):
        i, offset = 0, 1
        segs = []
        while i < len(target) - self.config.seq_len + 1:
            segs.append(target[i:i+self.config.seq_len])
            i += self.config.seq_len + offset
            offset += 1
            offset %= self.config.seq_len
        
        top_class = defaultdict(int)
        for target in segs:
            Cx1 = len(gzip.compress(target.encode()))
            distance_from_x1 = []
            for source, _ in self.sources:
                Cx2 = len(gzip.compress(source.encode()))
                x1x2 = ' '.join([target, source])
                Cx1x2 = len(gzip.compress(x1x2.encode()))
                ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
                distance_from_x1.append(ncd)
            sorted_idx = np.argsort(np.array(distance_from_x1))
            top_k_class = self.sources[sorted_idx[:self.config.n_windows], 1]
            for clazz in top_k_class:
                top_class[clazz] += 1
        predict_class = max(top_class, key=top_class.get)
        return predict_class
    

def cosine_similarity(x, y):
    dot_product = np.dot(x, y)

    # Calculate magnitudes of the vectors
    magnitude_x = np.linalg.norm(x)
    magnitude_y = np.linalg.norm(y)

    # Calculate cosine similarity
    return dot_product / (magnitude_x * magnitude_y)
    

class TokenLoopClassification(BaseClassification):

    def get_sources(self, sources):
        tokenizer = get_tokenizer()
        for source, label in tqdm(sources, desc='Load sources'):
            tokens = tokenizer.encode(source, add_special_tokens=False)
            for i in range(0, len(tokens)-self.config.seq_len+1, self.config.seq_len):
                self.sources.append([np.array(tokens[i:i+self.config.seq_len]), label])

    def classify(self, target: str):
        i, offset = 0, 1
        segs = []

        tokenizer = get_tokenizer()
        tokens = tokenizer.encode(target, add_special_tokens=False)

        while i < len(tokens) - self.config.seq_len + 1:
            segs.append(np.array(tokens[i:i+self.config.seq_len]))
            i += self.config.seq_len + offset
            offset += 1
            offset %= self.config.seq_len
        
        top_class = defaultdict(int)
        for target in segs:
            similarities = []
            for source, _ in self.sources:
                cs = -cosine_similarity(target, source) # less is better
                similarities.append(cs)
            sorted_idx = np.argsort(np.array(similarities))
            for idx in sorted_idx[:self.config.n_windows]:
                if similarities[idx] > self.config.threshold:
                    break
                top_class[self.sources[idx][1]] += 1
        if len(top_class) == 0:
            return 'unknown'
        predict_class = max(top_class, key=top_class.get)
        return predict_class