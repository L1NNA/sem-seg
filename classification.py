import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from torch import optim

from classification.load_classification import load_classification, ClassificationDataset
from classification.gzip_classification import GzipClassification, CompressLoopClassification, TokenLoopClassification
from baselines.cybertron import Cybertron, train_cybertron
from data_loader.setup_BPE import get_tokenizer
from utils.config import Config
from utils.checkpoint import load, save

def test_gzip_classification():
    config = Config()
    config.n_windows = 10
    config.seq_len = 2048
    config.threshold = 0.2
    data_path = './data2/train'

    training, segments = load_classification(data_path)

    gzip_classification = GzipClassification(config)
    gzip_classification.get_sources(segments)

    total = len(training)
    correct = 0
    for target, label in tqdm(training, desc='Classification'):
        predict_class = gzip_classification.classify(target)
        if label == predict_class:
            correct += 1

    print(f'Accuracy = {correct/total} out of {total}')


def collate_fn(batch):
    # Sort sequences by descending lengths
    xs_sorted, ys_sorted, labels_sorted = zip(*sorted(batch, key=lambda x: x[0].size(0), reverse=True))
    
    # Pad the sequences
    xs_padded = torch.nn.utils.rnn.pad_sequence(xs_sorted, batch_first=True)

     # Sort sequences by descending lengths
    xs_padded, ys_sorted, labels_sorted = zip(*sorted(zip(xs_padded, ys_sorted, labels_sorted), key=lambda x: x[1].size(0), reverse=True))
    
    # Pad the sequences
    ys_padded = torch.nn.utils.rnn.pad_sequence(ys_sorted, batch_first=True)

    # Convert labels to tensor (assuming they are scalar labels)
    xs_padded = torch.stack(xs_padded, dim=0)
    labels_tensor = torch.tensor(labels_sorted, dtype=torch.float32)
    
    return xs_padded, ys_padded, labels_tensor


def force_cybertron():
    vocab_size = get_tokenizer().vocab_size
    d_model = 512
    num_kernels = 32
    kernel_size = 16
    n_heads = 8
    n_layers = 6
    d_ff = 1024
    dropout = 0.1

    model = Cybertron(vocab_size, d_model, num_kernels, kernel_size, n_heads, n_layers, d_ff, dropout)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    train_dataset = ClassificationDataset('./data2', 'train')
    train_dataset.load_data()
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_dataset = ClassificationDataset('./data2', 'valid')
    val_dataset.load_data()
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    config = Config()
    config.checkpoint = './checkpoints'
    config.model_name = 'cybertron'
    config.epochs = 2
    config.device = torch.device('cuda:0')
    config.is_host = True
    model = model.to(config.device)

    init_epoch = load(config, model, optimizer, None)
    train_cybertron(model, train_loader, val_loader, config.epochs, config.device, optimizer, init_epoch)
    save(config, model, optimizer, None)


force_cybertron()