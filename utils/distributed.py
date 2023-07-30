import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from utils.config import Config


def setup_device(config:Config):
    if not config.gpu or not torch.cuda.is_available():
        config.device = torch.device("cpu")
        config.world_size = 0
        config.rank = 0
        config.is_host = True
        return
    elif config.local_rank > -1 or torch.cuda.device_count() == 1:
        config.world_size = 1
        config.rank = config.local_rank if config.local_rank > -1 else 0
        config.is_host = True
    else:
        dist.init_process_group(
            backend='nccl',
            init_method='env://'
        )

        config.rank = dist.get_rank()
        config.world_size = dist.get_world_size()
        config.is_host = config.rank == 0

    torch.cuda.set_device(config.rank)
    config.device = torch.device("cuda", config.rank)

def distribute_dataset(config:Config, dataset, batch_size):
    if dataset is None:
        return
    distributed = config.world_size > 1
    sampler = DistributedSampler(dataset, drop_last=True) \
        if distributed else None
    return DataLoader(dataset, num_workers=config.num_workers,
                      batch_size=batch_size,
                      shuffle=not distributed,
                      sampler=sampler)

def wrap_model(args:Config, model):
    model = model.to(args.device)
    if args.world_size > 1:
        model = DDP(
            model
        )
    return model