from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils.config import Config
from utils.metrics import confusion, calculate_auroc
from utils.checkpoint import save


def load_optimization(config:Config, model:nn.Module):
    # load training
    if config.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config.lr)
    elif config.optim == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.lr)

    return optimizer

def train(
        config:Config, model:nn.Module,
        train_loader:DataLoader, val_loader:DataLoader,
        optimizer:optim.Optimizer,
        writer:SummaryWriter,
        init_epoch:int=0,
    ):
    criterion = nn.CrossEntropyLoss()

    for epoch in range(init_epoch, config.epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        train_loss = []

        model.train()
        iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}') \
            if config.is_host else train_loader
        for x, y in iterator:

            # zero the parameter gradients
            optimizer.zero_grad()

            x = x.to(config.device)
            y:torch.Tensor = y.to(config.device)

            outputs = model(x)
            y = y.reshape(-1)

            loss = criterion(outputs, y)
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()
        
        stat = torch.tensor([np.sum(train_loss), len(train_loss)],
                dtype=torch.float32).to(config.device)
        if config.distributed:
            dist.reduce(stat, 0)
        if config.is_host:
            train_loss_result = stat[0] / stat[1]
            print("Epoch {0}: Train Loss: {1:.7f}" \
                .format(epoch + 1, train_loss_result))
            writer.add_scalar('Loss/Train', train_loss_result, epoch + 1)
        test(config, model, val_loader, 'Validation', writer, epoch + 1)
        if config.distributed:
            dist.barrier()
        train_loss.clear()

def train_coe(
        config:Config, model:nn.Module,
        train_loader:DataLoader, val_loader:DataLoader,
        optimizer:optim.Optimizer,
        writer:SummaryWriter,
        init_epoch:int=0
    ):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(init_epoch, config.epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        train_loss = []

        model.train()
        iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}') \
            if config.is_host else train_loader
        for x, y, y_mask, segs, labels in iterator:
            # zero the parameter gradients
            optimizer.zero_grad()

            x = x.to(config.device)
            y = y.to(config.device)
            y_mask = y_mask.to(config.device)
            segs = segs.to(config.device).reshape(-1)
            labels = labels.to(config.device).reshape(-1)

            segs_bar, labels_bar, similarity = model(x, None, y, y_mask)
            segs_bar = segs_bar.reshape(segs.size(0), -1)
            labels_bar = labels_bar.reshape(labels.size(0), -1)
            loss = 0.5 * criterion(segs_bar, segs) + criterion(labels_bar, labels) + similarity

            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        stat = torch.tensor([np.sum(train_loss), len(train_loss)],
                dtype=torch.float32).to(config.device)
        if config.distributed:
            dist.reduce(stat, 0)
        if config.is_host:
            train_loss_result = stat[0] / stat[1]
            print("Epoch {0}: Train Loss: {1:.7f}" \
                .format(epoch + 1, train_loss_result))
            # save(config, model, optimizer)
        test_coe(config, model, val_loader, 'Epoch {} validation'.format(epoch+1), writer, epoch)
        if config.distributed:
            dist.barrier()
        train_loss.clear()

def train_siamese(
        config:Config, model:nn.Module,
        train_loader:DataLoader, val_loader:DataLoader,
        optimizer:optim.Optimizer,
        writer:SummaryWriter,
        init_epoch:int=0
    ):
    criterion = nn.CosineEmbeddingLoss()
    for epoch in range(init_epoch, config.epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        train_loss = []

        model.train()
        iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}') \
            if config.is_host else train_loader
        for x, y, label in iterator:
            # zero the parameter gradients
            optimizer.zero_grad()

            x = x.to(config.device)
            y = y.to(config.device)
            label = label.to(config.device).squeeze()

            hx = model(x)
            hy = model(y)
            loss = criterion(hx, hy, label)

            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        stat = torch.tensor([np.sum(train_loss), len(train_loss)],
                dtype=torch.float32).to(config.device)
        if config.distributed:
            dist.reduce(stat, 0)
        if config.is_host:
            train_loss_result = stat[0] / stat[1]
            print("Epoch {0}: Train Loss: {1:.7f}" \
                .format(epoch + 1, train_loss_result))
            writer.add_scalar('Loss/Train', train_loss_result, epoch + 1)
        test_siamese(config, model, val_loader, 'Validation', writer, epoch+1)
        if config.distributed:
            dist.barrier()
        train_loss.clear()

def test(config:Config, model, test_loader, name, 
        writer:SummaryWriter, epoch):

    model.eval()
    
    labels = None
    logits = None
    predictions = None

    with torch.no_grad():
        desc = name + ':' + str(epoch)
        iterator = tqdm(test_loader, desc=desc) \
            if config.is_host else test_loader
        for x, y in iterator:
            x = x.to(config.device)
            y = y.to(config.device)

            outputs = model(x)

            y = y.reshape(-1)
            labels, logits, predictions = \
                _cache_logits(outputs, y, labels, logits, predictions)
    
    if config.distributed:
        labels, logits, predictions = _reduce_metrics(labels, logits, predictions, config)
    else:
        labels, logits, predictions = labels.cpu(), logits.cpu(), predictions.cpu()

    if config.is_host:
        _print_metrics(labels, logits, predictions, config, name, writer, epoch)

def test_coe(config:Config, model, test_loader, name, 
        writer:SummaryWriter, epoch):

    model.eval()
    
    labels_true = None
    labels_logits = None
    labels_predictions = None

    segs_true = None
    segs_logits = None
    segs_predictions = None

    with torch.no_grad():
        iterator = tqdm(test_loader, desc=name) \
            if config.is_host else test_loader
        for x, y, y_mask, segs, labels in iterator:
            x = x.to(config.device)
            segs = segs.to(config.device).reshape(-1)
            labels = labels.to(config.device).reshape(-1)

            segs_bar, labels_bar, z = model.module.inference(x, None) if config.distributed else model.inference(x, None)
            segs_bar = segs_bar.reshape(segs.size(0), -1)
            labels_bar = labels_bar.reshape(labels.size(0), -1)

            segs_true, segs_logits, segs_predictions = \
                _cache_logits(segs_bar, segs, segs_true, segs_logits, segs_predictions)
            labels_true, labels_logits, labels_predictions = \
                _cache_logits(labels_bar, labels, labels_true, labels_logits, labels_predictions)

    
    if config.distributed:
        segs_true, segs_logits, segs_predictions = \
            _reduce_metrics(segs_true, segs_logits, segs_predictions, config)
        labels_true, labels_logits, labels_predictions = \
            _reduce_metrics(labels_true, labels_logits, labels_predictions, config)

    if config.is_host:
        _print_metrics(segs_true, segs_logits, segs_predictions, config, name + ' segmentation')
        _print_metrics(labels_true, labels_logits, labels_predictions, config, name + ' labeling')

def _cache_logits(bar, ground_truth, truths, logits, predictions):
    probs = torch.softmax(bar, dim=1) # b x o
    predicted = torch.argmax(probs, dim=1) # b x 1

    truths = ground_truth if truths is None else torch.cat((truths, ground_truth))
    logits = probs if logits is None else torch.cat([logits, probs])
    predictions = predicted if predictions is None \
        else torch.cat((predictions, predicted))
    return truths, logits, predictions

def  _reduce_metrics(truths, logits, predictions, config):
    int_stats = [
        torch.zeros(2, truths.size(0), dtype=torch.long).to(config.device) \
        for _ in range(config.world_size)
    ] if config.is_host else None
    float_stats = [
        torch.zeros(logits.size(), dtype=torch.float32).to(config.device) \
        for _ in range(config.world_size)
    ] if config.is_host else None
    int_values = torch.stack((predictions, truths)).to(config.device)
    logits = logits.to(config.device)
    dist.gather(int_values, int_stats, 0)
    dist.gather(logits, float_stats, 0)

    if config.is_host:
        stat = torch.cat(int_stats, dim=1)
        predictions = stat[0].cpu()
        truths = stat[1].cpu()
        logits = torch.cat(float_stats).cpu()

    return truths, logits, predictions

def _print_metrics(truths, logits, predictions, config, name, writer, epoch):
    total = truths.size(0)
    accuracy, precision, recall, f1 = confusion(truths, predictions, logits.size(1))
    auroc = calculate_auroc(truths, logits)

    print("{}:{}: Accuracy: {:.2f}%, Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}, AUCROC: {:.2f}, Total: {}"\
            .format(name, epoch, accuracy, precision, recall, f1, auroc, total))
    writer.add_scalar('Accuracy/'+name, accuracy, epoch)
    writer.add_scalar('Precision/'+name, precision, epoch)
    writer.add_scalar('Recall/'+name, recall, epoch)
    writer.add_scalar('F1/'+name, f1, epoch)
    writer.add_scalar('AUROC/'+name, auroc, epoch)

def test_siamese(config:Config, model, test_loader, name, 
        writer:SummaryWriter, epoch):

    model.eval()
    
    ground_truth = None
    logits = None
    predictions = None

    with torch.no_grad():
        iterator = tqdm(test_loader, desc=name) \
            if config.is_host else test_loader
        for x, y, labels in iterator:
            x = x.to(config.device)
            y = y.to(config.device)
            labels = labels.to(config.device).reshape(-1)

            hx = model(x)
            hy = model(y)
            cosine_similarity = F.cosine_similarity(hx, hy, dim=1)

            outputs = (cosine_similarity + 1) / 2 # b x 1
            outputs = outputs.unsqueeze(-1)
            outputs = torch.concat([1-outputs, outputs], dim=1)
            
            ground_truth, logits, predictions = \
                _cache_logits(outputs, labels, ground_truth, logits, predictions)
    
    if config.distributed:
        ground_truth, logits, predictions = \
            _reduce_metrics(ground_truth, logits, predictions, config)
    else:
        ground_truth, logits, predictions = ground_truth.cpu(), logits.cpu(), predictions.cpu()

    if config.is_host:
        _print_metrics(ground_truth, logits, predictions, config, name + ' clone', writer, epoch)