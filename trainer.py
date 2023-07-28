from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from utils.config import Config


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
        for x, y in tqdm(train_loader, desc=f'Epoch {epoch}'):

            # zero the parameter gradients
            optimizer.zero_grad()

            x = x.to(config.device)
            y:torch.Tensor = y.to(config.device)

            if config.model == 'cosFormer':
                outputs, _ = model(x, None)
            else:
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
        train_loss.clear()


def validation(config, model, val_loader, train_loss, epoch):
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
    if not config.distributed or config.rank == 0:
        accuracy = 100 * stat[0] / stat[1]
        train_loss = stat[2] / stat[3]
        print("Epoch: {0} | Train Loss: {1:.7f} Accuracy: {2:.2f}%" \
                .format(epoch + 1, train_loss, accuracy))


def test(config, model, test_loader):

    model.eval()
    correct = 0
    total = 0
    for x, y in test_loader:
        x = x.to(config.device)
        y = y.to(config.device)

        outputs = model(x)
        if config.model == 'cats':
            outputs = outputs[0]
        outputs = outputs[:, -1, :]
        y = y[:, -1]
        predicted = torch.argmax(
            torch.softmax(outputs, dim=1), dim=1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    
    stat = torch.tensor([correct, total], dtype=torch.long).to(config.device)
    dist.reduce(stat, 0)
    if not config.distributed or config.rank == 0:
        accuracy = 100 * stat[0] / stat[1]
        print("Accuracy: {:.2f}%".format(accuracy))
