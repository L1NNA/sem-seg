import os
import torch
from collections import OrderedDict

from utils.config import Config



def load_checkpoint(path, model, optimizer, scheduler, config):
    # the model is saved from gpu0 so we need to map it to CPU first
    state = torch.load(path, map_location=lambda storage, _: storage)
    init_epoch = state["epoch"]
    if config.distributed:
        model.load_state_dict(state["model"])
    else:
        new_state_dict = OrderedDict()
        for k, v in state["model"].items():
            name = k[7:] if k.startswith('module.') else k  # remove `module.` prefix
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if "scheduler_epoch" in state and scheduler is not None:
        scheduler.step(state["scheduler_epoch"])
    return init_epoch

def load(config:Config, model, optimizer, scheduler):
    init_epoch = 0
    checkpoint = os.path.join(config.checkpoint, config.model_name + '.pt')
    if os.path.exists(checkpoint):
        try:
            init_epoch = load_checkpoint(
                checkpoint, model, optimizer, scheduler, config
            )
            if config.is_host:
                print('Checkpoint loaded')
        except Exception as e:
            if config.is_host:
                print('Failed to load checkpoint for', model, e)
    return init_epoch

def save(config:Config, model, optimizer, scheduler):
    if not os.path.exists(config.checkpoint):
        return
    checkpoint = os.path.join(config.checkpoint, config.model_name + '.pt')
    state = dict()
    state["epoch"] = config.epochs
    state["model"] = model.state_dict()
    state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_epoch"] = scheduler.last_epoch
    torch.save(state, checkpoint)
    if config.is_host:
        print('Checkpoint saved')