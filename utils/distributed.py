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
        return
    elif config.local_rank > -1 or torch.cuda.device_count() == 1:
        config.world_size = 1
        config.rank = config.local_rank if config.local_rank > -1 else 0
    else:
        dist.init_process_group(
            backend='nccl',
            init_method='env://'
        )

        config.rank = dist.get_rank()
        config.world_size = dist.get_world_size()

    torch.cuda.set_device(config.rank)
    config.device = torch.device("cuda", config.rank)

def distribute_dataset(config:Config, dataset):
    is_distributed = config.world_size > 1
    sampler = DistributedSampler(dataset) if is_distributed else None
    dataloader = DataLoader(dataset, num_workers=config.num_workers,
                            batch_size=config.batch_size,
                            shuffle=not is_distributed,
                            sampler=sampler)
    return dataloader

def wrap_model(args:Config, model):
    model = model.to(args.device)
    if args.world_size > 1:
        model = DDP(
            model
        )
    return model