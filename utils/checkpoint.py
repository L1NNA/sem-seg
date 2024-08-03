import os
import torch
from collections import OrderedDict

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn

from utils.config import Config


def load_checkpoint(path, model, optimizer, config):
    state = torch.load(path)
    init_epoch = state["epoch"]
    
    new_state_dict = OrderedDict() # state["model"]
    for k, v in state["model"].items():
        name = k[7:] if k.startswith('module.') else k  # remove `module.` prefix
        new_state_dict[name] = v

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    del state
    return init_epoch

def load(config:Config, model, optimizer):
    init_epoch = 0
    checkpoint = os.path.join(config.checkpoint, config.model_name + '.pt')
    if os.path.exists(checkpoint):
        try:
            init_epoch = load_checkpoint(
                checkpoint, model, optimizer, config
            )
            if config.is_host:
                print('Checkpoint loaded')
        except Exception as e:
            if config.is_host:
                print('Failed to load checkpoint for', model, e)
    return init_epoch

def save(config:Config, model, optimizer):
    if not os.path.exists(config.checkpoint):
        return
    checkpoint = os.path.join(config.checkpoint, config.model_name + '.pt')
    state = dict()
    state["epoch"] = config.epochs
    state["model"] = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    state["optimizer"] = optimizer.state_dict()
    torch.save(state, checkpoint)
    if config.is_host:
        print('Checkpoint saved')