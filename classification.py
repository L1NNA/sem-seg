import numpy as np

from classification.load_classification import load_classification
from classification.gzip_classification import GzipClassification, CompressLoopClassification, TokenLoopClassification

from utils.config import Config

config = Config()
config.n_windows = 3
config.seq_len = 3
config.threshold = 0.2
data_path = './data2/train'

training, segments = load_classification(data_path)

gzip_classification = TokenLoopClassification(config)
gzip_classification.get_sources(segments)

total = len(training)
correct = 0
for target, label in training:
    predict_class = gzip_classification.classify(target)
    if label == predict_class:
        correct += 1

print(f'Accuracy = {correct/total}')