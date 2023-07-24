import torch
import torch.nn as nn
from torch import optim
from typing import Optional
import numpy as np

from torch.utils.data import DataLoader

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


def sequential_training(config:Config, model:nn.Module,
                        train_loader:DataLoader, val_loader:DataLoader,
                        optimizer:optim.Optimizer,
                        scheduler:Optional[optim.lr_scheduler._LRScheduler],
                        init_epoch:int=0
                        ):
    # predict next token
    criterion = nn.CrossEntropyLoss()

    for epoch in range(init_epoch, config.epochs):
        train_loss = []

        model.train()
        for i, (x, y) in enumerate(train_loader):

            # zero the parameter gradients
            optimizer.zero_grad()

            x = x.to(config.device)
            y:torch.Tensor = y.to(config.device)

            if config.model == 'cats':
                outputs, aux_loss = model(x)
            else:
                outputs = model(x)
            
            outputs = outputs.reshape(-1, outputs.shape[-1])
            y = y.reshape(-1)

            loss = criterion(outputs, y)
            if config.model == 'cats':
                loss += aux_loss
            train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                print("epoch: {0} | item: {1} | loss: {2:.7f}".format(epoch + 1, i + 1, loss.item()))
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()


        # Validation

        model.eval()
        correct = 0
        total = 0
        for x, y in val_loader:
            x = x.to(config.device)
            y = y.to(config.device)

            outputs = model(x)
            if config.model == 'cats':
                outputs = outputs[0]
            outputs = outputs[:, -1, :]
            predicted = torch.argmax(outputs, dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        
        train_loss = np.average(train_loss)
        print("Epoch: {0} | Train Loss: {1:.7f} Accuracy: {2:.2f}%".format(epoch + 1, train_loss, 100 * correct / total))
