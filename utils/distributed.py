import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


def setup(args):

    dist.init_process_group(
        backend='nccl',
        init_method='env://'
    )

    args.rank = torch.distributed.get_rank()
    args.world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device("cuda", args.rank)


def sample_dataset(args, dataset):
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    return dataloader

def wrap_model(args, model):
    model = model.to(args.device)
    model = DDP(
        model,
        device_ids=[args.local_rank]
    )
    return model