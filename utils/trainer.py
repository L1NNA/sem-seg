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
        test(config, model, val_loader, 'Epoch {} validation'.format(epoch+1))
        if config.distributed:
            dist.barrier()
        train_loss.clear()


def validation(config:Config, model, val_loader, train_loss, epoch):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(config.device)
            y = y.to(config.device)

            outputs = model(x)
            
            y = y[:, -1]
            predicted = torch.argmax(
                torch.softmax(outputs, dim=1), dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    
    stat = torch.tensor([correct, total, np.sum(train_loss), len(train_loss)],
                dtype=torch.float32).to(config.device)
    if config.distributed:
        dist.reduce(stat, 0)
    if config.is_host:
        accuracy = 100 * stat[0] / stat[1]
        train_loss = stat[2] / stat[3]
        print("Epoch {0}: Train Loss: {1:.7f} Accuracy: {2:.2f}%" \
                .format(epoch + 1, train_loss, accuracy))


def test(config:Config, model, test_loader, name):

    model.eval()
    
    labels = None
    logits = None
    predictions = None

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(config.device)
            y = y.to(config.device)

            outputs = model(x)

            y = y[:, -1]
            probs = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(probs, dim=1)

            y = y.detach().cpu()
            probs = probs.detach().cpu()
            predicted = predicted.detach().cpu()

            labels = y if labels is None else torch.cat((labels, y))
            logits = probs if logits is None else torch.cat([logits, probs])
            predictions = predicted if predictions is None \
                else torch.cat((predictions, predicted))
    
    if config.distributed:
        int_stats = [
        torch.zeros(2, labels.size(0), dtype=torch.long).to(config.device) \
            for _ in range(config.world_size)
        ] if config.is_host else None
        float_stats = [
            torch.zeros(logits.size(), dtype=torch.float32).to(config.device) \
            for _ in range(config.world_size)
        ] if config.is_host else None
        int_values = torch.stack((predictions, labels)).to(config.device)
        logits = logits.to(config.device)
        dist.gather(int_values, int_stats, 0)
        dist.gather(logits, float_stats, 0)

        if config.is_host:
            stat = torch.cat(int_stats, dim=1)
            predictions = stat[0].cpu()
            labels = stat[1].cpu()
            logits = torch.cat(float_stats).cpu()

    if config.is_host:
        total = labels.size(0)
        accuracy, precision, recall, f1 = confusion(labels, predictions, logits.size(1))
        auroc = calculate_auroc(labels, logits)

        print("{}: Accuracy: {:.2f}%, Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}, AUCROC: {:.2f}, Total: {}"\
              .format(name, accuracy, precision, recall, f1, auroc, total))

