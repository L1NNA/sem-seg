import os
import shutil
import torch

from utils.config import Config


def load_path(path):
    print("loading from " + path)
    # the model is saved from gpu0 so we need to map it to CPU first
    return torch.load(path, map_location=lambda storage, _: storage)


def load_checkpoint(args, path, model, optimizer, scheduler):
    f = load_path(path)
    ep_init = f["epoch"]
    model.load_state_dict(f["model"])
    optimizer.load_state_dict(f["optimizer"])
    if "scheduler_epoch" in f:
        scheduler.step(f["scheduler_epoch"])

    return ep_init


def load(args:Config, model, optimizer, scheduler):
    ep_init = 0
    if args.checkpoint != "" and os.path.exists(args.checkpoint):
        try:
            ep_init = load_checkpoint(
                args, args.checkpoint, model, optimizer, scheduler
            )
        except Exception as e:
            print(f"load failed: {e}")
            # # try the backup checkpoint
            # if os.path.exists(args.checkpoint + ".bak"):
            #     try:
            #         ep_init = load_checkpoint(
            #             args,
            #             args.checkpoint + ".bak",
            #             model,
            #             optimizer,
            #             scheduler,
            #         )
            #     except Exception as e:
            #         print(f"backup load failed: {e}")
    return ep_init


def save(args:Config, model, optimizer, scheduler):
    if args.checkpoint != "":
        if os.path.exists(args.checkpoint):
            if (
                args.checkpoint_freq > 0
                and args.ep > 0
                and args.ep % args.checkpoint_freq == 0
            ):
                try:
                    shutil.copyfile(
                        args.checkpoint, args.checkpoint + "." + str(args.ep)
                    )
                except:
                    print("save copy failed")
            # make a backup in case this save fails
            os.replace(args.checkpoint, args.checkpoint + ".bak")
        f = dict()
        f["epoch"] = args.ep + 1
        f["model"] = model.state_dict()
        f["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            f["scheduler_epoch"] = scheduler.last_epoch
        torch.save(f, args.checkpoint)