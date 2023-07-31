from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from utils.config import Config
from utils.metrics import confusion, calculate_auroc


def load_optimization(config:Config, model):
    # load training
    if config.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config.lr)
    elif config.optim == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.lr)

    if config.lr_decay:
        # will do warm-up manually later
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, config.epochs * config.batch_size
        )
    elif config.lr_warmup > 0:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda ep: min(1, ep / config.lr_warmup)
        )
    else:
        scheduler = None

    return optimizer, scheduler


def train(
        config:Config, model:nn.Module,
        train_loader:DataLoader, val_loader:DataLoader,
        optimizer:optim.Optimizer,
        scheduler:Optional[optim.lr_scheduler._LRScheduler],
        init_epoch:int=0
    ):
    criterion = nn.CrossEntropyLoss()

    for epoch in range(init_epoch, config.epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        train_loss = []

        model.train()
        iterator = tqdm(train_loader, desc=f'Epoch {epoch}') \
            if config.is_host else train_loader
        for x, y in iterator:

            # zero the parameter gradients
            optimizer.zero_grad()

            x = x.to(config.device)
            y:torch.Tensor = y.to(config.device)

            outputs = model(x)
            
            if config.data == 'binary':
                if config.model == 'cats':
                    outputs = torch.mean(outputs, dim=1)
                else:
                    outputs = outputs[:, -1, :]
            outputs = outputs.reshape(-1, outputs.size(-1))
            y = y.reshape(-1)

            loss = criterion(outputs, y)
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        validation(config, model, val_loader, train_loss, epoch)
        dist.barrier()
        train_loss.clear()


def validation(config:Config, model, val_loader, train_loss, epoch):
    model.eval()
    correct = 0
    total = 0
    for x, y in val_loader:
        x = x.to(config.device)
        y = y.to(config.device)

        outputs = model(x)
        
        if config.model == 'cats':
            outputs = torch.mean(outputs, dim=1)
        else:
            outputs = outputs[:, -1, :]
        y = y[:, -1]
        predicted = torch.argmax(
            torch.softmax(outputs, dim=1), dim=1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    
    
    stat = torch.tensor([correct, total, np.sum(train_loss), len(train_loss)],
                dtype=torch.float32).to(config.device)
    dist.reduce(stat, 0)
    if config.is_host:
        accuracy = 100 * stat[0] / stat[1]
        train_loss = stat[2] / stat[3]
        print("Epoch: {0} | Train Loss: {1:.7f} Accuracy: {2:.2f}%" \
                .format(epoch + 1, train_loss, accuracy))


def test(config:Config, model, test_loader):

    model.eval()
    
    labels = None
    logits = None
    predictions = None

    for x, y in test_loader:
        x = x.to(config.device)
        y = y.to(config.device)

        outputs = model(x)
        
        if config.model == 'cats':
            outputs = torch.mean(outputs, dim=1)
        else:
            outputs = outputs[:, -1, :]

        y = y[:, -1]
        probs = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(probs, dim=1)

        y = y.detach().cpu()
        probs = probs.detach().cpu()[:, 1]
        predicted = predicted.detach().cpu()

        labels = y if labels is None else torch.cat((labels, y))
        logits = probs if logits is None else torch.cat([logits, probs])
        predictions = predicted if predictions is None \
            else torch.cat((predictions, predicted))
    
    
    
    int_stats = [
        torch.zeros(2, labels.size(0), dtype=torch.long).to(config.device) \
        for _ in range(config.world_size)
    ] if config.is_host else None
    float_stats = [
        torch.zeros(logits.size(0), dtype=torch.float32).to(config.device) \
        for _ in range(config.world_size)
    ] if config.is_host else None
    int_values = torch.stack((predictions, labels)).to(config.device)
    logits = logits.to(config.device)
    dist.gather(int_values, int_stats, 0)
    dist.gather(logits, float_stats, 0)

    if config.is_host:
        stat = torch.cat(int_stats, dim=1)
        predictions = stat[0]
        labels = stat[1]
        logits = torch.cat(float_stats)

        total = labels.size(0) * config.world_size
        accuracy, recall, precision = confusion(labels, predictions)
        auroc = calculate_auroc(labels, logits)

        print("Accuracy: {:.2f}%, Precision: {:.2f}, Recall: {:.2f}, AUCROC: {:2f}, Total: {}"\
              .format(accuracy, precision, recall, auroc, total))

