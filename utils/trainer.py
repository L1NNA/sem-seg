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
from utils.metrics import confusion, calculate_auroc, calculate_mrr
from utils.checkpoint import save
from utils.distributed import gather_tensors

from layers.losses import SegInfoNCELoss


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
        init_epoch:int=0
    ):
    ce_criterion = None
    sim_criterion = None
    is_coe = config.model == 'coe'
    if config.do_seg or config.do_cls:
        ce_criterion = nn.CrossEntropyLoss()
    if config.do_ccr:
        ccr_criterion = SegInfoNCELoss(config.n_windows)

    for epoch in range(init_epoch, config.epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        train_loss = []

        model.train()
        iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}') \
            if config.is_host else train_loader
        for x, x_mask, segs, labels, y, y_mask in iterator:
            # zero the parameter gradients
            optimizer.zero_grad()
            
            x = x.to(config.device)
            x_mask = x_mask.to(config.device)
            if config.do_ccr and is_coe:
                y = y.to(config.device)
                y_mask = y_mask.to(config.device)
                outputs = model(x, x_mask, y, y_mask, True)
            else:
                outputs = model(x, x_mask)

            loss = 0
            if config.do_seg:
                segs = segs.to(config.device).reshape(-1)
                seg_preds = outputs[0] if is_coe else outputs
                loss += ce_criterion(seg_preds, segs)

            if config.do_cls:
                labels = labels.to(config.device).reshape(-1)
                cls_preds = outputs[1] if is_coe else outputs
                loss += ce_criterion(cls_preds, labels)

            if config.do_ccr:
                if is_coe:
                    loss += torch.mean(outputs[3])
                else:
                    y = y.to(config.device)
                    y_mask = y_mask.to(config.device)
                    x_embed = outputs[2] if is_coe else outputs
                    y_embed = model(y, y_mask)
                    loss += ccr_criterion(x_embed, y_embed)

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        
        loss = np.mean(train_loss)
        print("Epoch {0}: Train Loss: {1:.7f}".format(epoch + 1, loss))
        writer.add_scalar('Train Loss', loss, epoch + 1)
        save(config, model, optimizer)
        test(config, model, val_loader, 'Validation', writer, epoch+1)
        train_loss.clear()

@torch.no_grad()
def test(config:Config, model, test_loader, name, 
        writer:SummaryWriter, epoch):
    epsilon = 1e-8

    model.eval()

    is_coe = config.model == 'coe'

    seg_preds, seg_true = None, None
    cls_preds, cls_true = None, None
    x_embeds, y_embeds = None, None
    
    index = 0
    for x, x_mask, segs, labels, y, y_mask in tqdm(test_loader, desc=name):
        x = x.to(config.device)
        x_mask = x_mask.to(config.device)
        # outputs = model(x, x_mask)
        if config.do_ccr and is_coe:
            y = y.to(config.device)
            y_mask = y_mask.to(config.device)
            outputs = model(x, x_mask, y, y_mask, False)
        else:
            outputs = model(x, x_mask)

        if config.do_seg:
            segs = segs.reshape(-1)
            seg_outputs = outputs[0] if is_coe else outputs
            probs = torch.softmax(seg_outputs, dim=1) # b x o
            preds = torch.argmax(probs, dim=1).cpu() # b

            seg_true = segs if seg_true is None else torch.cat((seg_true, segs))
            seg_preds = preds if seg_preds is None else torch.cat((seg_preds, preds))

        if config.do_cls:
            labels = labels.reshape(-1)
            cls_outputs = outputs[1] if is_coe else outputs
            probs = torch.softmax(cls_outputs, dim=1) # b x o
            preds = torch.argmax(probs, dim=1).cpu() # b

            cls_true = labels if cls_true is None else torch.cat((cls_true, labels))
            cls_preds = preds if cls_preds is None else torch.cat((cls_preds, preds))

        if config.do_ccr:
            x_embed = outputs[2] if is_coe else outputs # b x d
            if is_coe:
                y_embed = outputs[3]
            else:
                y = y.to(config.device)
                y_mask = y_mask.to(config.device)
                y_embed = model(y, y_mask)
            
            x_embed = x_embed / (x_embed.norm(dim=1, keepdim=True) + epsilon)
            y_embed = y_embed / (y_embed.norm(dim=1, keepdim=True) + epsilon)
            x_embed = x_embed.cpu()
            y_embed = y_embed.cpu()
            x_embeds = x_embed if x_embeds is None else torch.cat([x_embeds, x_embed], dim=0)
            y_embeds = y_embed if y_embeds is None else torch.cat([y_embeds, y_embed], dim=0)
            if not config.do_cls:
                cls_true = labels if cls_true is None else torch.cat((cls_true, labels))

    if config.do_seg:
        _print_metrics(seg_true, None, seg_preds, config, name + ' segmentation', writer, epoch)
    if config.do_cls:
        _print_metrics(cls_true, None, cls_preds, config, name + ' labeling', writer, epoch)
    if config.do_ccr:
        code_clone_retrieval(x_embeds, y_embeds, cls_true, config, name, writer, epoch)

def code_clone_retrieval(x_embeds, y_embeds, masking, config, name, writer, epoch):
    name = name + ' retrieval'
    x_embeds = x_embeds #.to(config.device)
    y_embeds = y_embeds #.to(config.device)
    total = x_embeds.size(0)
    weights = torch.matmul(x_embeds, y_embeds.transpose(0, 1)) #.cpu()
    mmr = calculate_mrr(weights, masking)
    print("{}:{}: MMR: {:.2f}, Total: {}".format(name + ' retrieval', epoch, mmr, total))
    writer.add_scalar('MMR/'+name, mmr, epoch)

def _print_metrics(truths, weights, predictions, config, name, writer, epoch):
    total = truths.size(0)
    accuracy, precision, recall, f1 = confusion(truths, predictions)
    # auroc = calculate_auroc(truths, weights)

    print("{}:{}: Accuracy: {:.2f}%, Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}, Total: {}"\
            .format(name, epoch, accuracy, precision, recall, f1, total))
    writer.add_scalar('Accuracy/'+name, accuracy, epoch)
    writer.add_scalar('Precision/'+name, precision, epoch)
    writer.add_scalar('Recall/'+name, recall, epoch)
    writer.add_scalar('F1/'+name, f1, epoch)
    writer.add_scalar('Total/'+name, total, epoch)
    # writer.add_scalar('AUROC/'+name, auroc, epoch)

