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
from utils.clone_search import CloneSearch
from utils.distributed import gather_tensors


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
            # masking = masking.to(config.device)
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
    sim_criterion = nn.CosineEmbeddingLoss()
    for epoch in range(init_epoch, config.epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        train_loss = []

        model.train()
        iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}') \
            if config.is_host else train_loader
        for x, masking, segs, labels in iterator:
            # zero the parameter gradients
            optimizer.zero_grad()

            x = x.to(config.device)
            masking = masking.to(config.device)
            segs = segs.to(config.device).reshape(-1)
            labels = labels.to(config.device).reshape(-1)
            sims = torch.ones_like(labels).to(config.device)

            segs_bar, labels_bar, hx = model(x, None)
            hy = model.module.representation(x, masking) if config.distributed else model.representation(x, masking)

            segs_bar = segs_bar.reshape(segs.size(0), -1)
            labels_bar = labels_bar.reshape(labels.size(0), -1)
            hx = hx.reshape(sims.size(0), -1)
            hy = hy.reshape(sims.size(0), -1)

            loss = criterion(segs_bar, segs) + criterion(labels_bar, labels) + sim_criterion(hx, hy, sims)

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
            save(config, model, optimizer)
        test_coe(config, model, val_loader, 'Validation', writer, epoch+1)
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
            # masking = masking.to(config.device)
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

    segs_cache = None
    segs_weights_cache = None
    segs_bar_cache = None
    
    labels_cache = None
    labels_weights_cahe = None
    labels_bar_cache = None

    sims_cache = None
    sim_bar_cache = None

    with torch.no_grad():
        cs = CloneSearch(config)
        cs.load_sources(test_loader.dataset, model)
        iterator = tqdm(test_loader, desc=name) \
            if config.is_host else test_loader
        for x, segs, labels, sims in iterator:
            x = x.to(config.device)
            segs = segs.to(config.device).reshape(-1)
            labels = labels.to(config.device).reshape(-1)
            sims = sims.to(config.device).reshape(-1)

            segs_logits, labels_logits, hx = model(x, None)
            segs_logits = segs_logits.reshape(segs.size(0), -1)
            labels_logits = labels_logits.reshape(labels.size(0), -1)
            hx = hx.reshape(sims.size(0), -1)
            sims_logits = cs.clone_search(hx)

            segs_cache, segs_weights_cache, segs_bar_cache = \
                _cache_logits(segs_logits, segs, segs_cache, segs_weights_cache, segs_bar_cache)
            labels_cache, labels_weights_cahe, labels_bar_cache = \
                _cache_logits(labels_logits, labels, labels_cache, labels_weights_cahe, labels_bar_cache)
            sims_cache, _, sim_bar_cache = \
                _cache_logits(sims_logits, sims, sims_cache, None, sim_bar_cache)

    if config.distributed:
        segs_cache, segs_weights_cache, segs_bar_cache = \
            _reduce_metrics(segs_cache, segs_weights_cache, segs_bar_cache, config)
        labels_cache, labels_weights_cahe, labels_bar_cache = \
            _reduce_metrics(labels_cache, labels_weights_cahe, labels_bar_cache, config)
        sims_cache, _, sim_bar_cache = \
            _reduce_metrics(sims_cache, None, sim_bar_cache, config)

    if config.is_host:
        _print_metrics(segs_cache, segs_weights_cache, segs_bar_cache, config, name + ' segmentation', writer, epoch)
        _print_metrics(labels_cache, labels_weights_cahe, labels_bar_cache, config, name + ' labeling', writer, epoch)
        acc = (sims_cache == sim_bar_cache).sum().item()
        total = sims_cache.size(0)
        print("Clone Search:{}: Accuracy: {:.2f}% Total: {}".format(epoch, acc/total, total))

def _cache_logits(logits, ground_truth, gt_cache, weights_cache, bar_cache):
    weights = torch.softmax(logits, dim=1) # b x o
    bar = torch.argmax(weights, dim=1) # b x 1

    gt_cache = ground_truth if gt_cache is None else torch.cat((gt_cache, ground_truth))
    weights_cache = weights if weights_cache is None else torch.cat([weights_cache, weights])
    bar_cache = bar if bar_cache is None else torch.cat((bar_cache, bar))
    return gt_cache, weights_cache, bar_cache

def  _reduce_metrics(gt_cache, weights_cache, bar_cache, config):
    gt_cache = gather_tensors(gt_cache, config)
    weights_cache = gather_tensors(weights_cache, config)
    bar_cache = gather_tensors(bar_cache, config)
    return gt_cache, weights_cache, bar_cache

def _print_metrics(truths, weights, predictions, config, name, writer, epoch):
    total = truths.size(0)
    accuracy, precision, recall, f1 = confusion(truths, predictions, weights.size(1))
    auroc = calculate_auroc(truths, weights)

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