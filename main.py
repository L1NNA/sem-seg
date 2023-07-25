import argparse
import random

import numpy as np
import torch

from baselines import cats, cosformer, graphcodebert, transformer
from data_loader import load_dataset, load_tokenizer
from trainer import load_optimization, sequential_training
from utils.checkpoint import load, save
from utils.config import Config
from utils.distributed import distribute_dataset, setup_device, wrap_model


def define_argparser():
    parser = argparse.ArgumentParser(description='')

    # loading model
    parser.add_argument('--training', action='store_true', help='status')
    parser.add_argument('--model', required=True, default='transformer',
                        choices=['transformer', 'cosFormer', 'cats', 'graphcodebert'],
                        help='model name, options: [TBD]')
    parser.add_argument('--model_name', type=str, required=True,
                        help='the name of weight files')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint',
                        help='checkpoint path')

    # data_loader
    parser.add_argument('--data_path', type=str, default='./data',
                        help='the path of data files')
    parser.add_argument('--data', required=True, default='seq',
                        choices=['seq', 'binary'], help='datasaets')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size of train input data')
    parser.add_argument('--seq_len', type=int, default=512,
                        help='input sequence length')
    parser.add_argument('--mem_len', type=int, default=1024,
                        help='memory sequence length')

    # hyper-parameters
    transformer.add_args(parser)
    cats.add_args(parser)
    cosformer.add_args(parser)
    graphcodebert.add_args(parser)

    # optimization
    # parser.add_argument('--itr', type=int, default=2,
    #                     help='experiments times')
    parser.add_argument('--epochs', type=int, default=10,
                        help='train epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='optimizer learning rate')
    # parser.add_argument('--patience', type=int, default=3,
    #                     help='early stopping patience')
    parser.add_argument("--optim", type=str, default="adam",
                        choices=("sgd", "adam"),
                        help="optimization method",
    )
    parser.add_argument("--lr_warmup", type=int, default=0,
        help="linearly increase LR from 0 during K updates",
    )
    parser.add_argument("--lr_decay", action="store_true", default=False,
        help="decay learning rate with cosine scheduler",
    )

    # distribution
    parser.add_argument('--gpu', action='store_true', help='use gpu or not')
    parser.add_argument('--local_rank', type=int, default=-1,
        help="local instance",
    )
    
    return parser


def main(arg=None):
    parser = define_argparser()
    config:Config = parser.parse_args(arg)

    # set random seed
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # load tokenizer
    load_tokenizer(config)

    # load model
    models = {
        'transformer': transformer.Transformer,
        'cosFormer': cosformer.Cosformer,
        'cats': cats.CATS,
        'graphcodebert': graphcodebert.GraphCodeBERT,
    }

    setup_device(config)
    model = models[config.model](config)
    model = wrap_model(config, model)

    # load dataset
    dataset = load_dataset(config)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = distribute_dataset(config, train_dataset)
    val_loader = distribute_dataset(config, val_dataset)
    
    # load optimization
    optimizer, scheduler = load_optimization(config, model)

    # load models
    ep_init = load(config, model, optimizer, scheduler)
    if config.training:
        sequential_training(config, model, train_loader, val_loader,
                            optimizer, scheduler, ep_init)
        # if config.rank == 0:
        #     save(config, model, optimizer, None)

    # TODO: test

if __name__ == "__main__":
    main()