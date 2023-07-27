import os
import torch

from utils.config import Config


def load_checkpoint(path, model, optimizer, scheduler):
    # the model is saved from gpu0 so we need to map it to CPU first
    state = torch.load(path, map_location=lambda storage, _: storage)
    init_epoch = state["epoch"]
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if "scheduler_epoch" in state:
        scheduler.step(state["scheduler_epoch"])
    return init_epoch

def load(config:Config, model, optimizer, scheduler):
    init_epoch = 0
    checkpoint = os.path.join(config.checkpoint, config.model_name + '.pt')
    if os.path.exists(checkpoint):
        try:
            init_epoch = load_checkpoint(
                checkpoint, model, optimizer, scheduler
            )
            print('Checkpoint loaded')
        except Exception as e:
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
    print('Checkpoint saved')