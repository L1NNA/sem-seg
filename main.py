import argparse
import random

import numpy as np
import torch

from baselines import cats, cosformer, graphcodebert, transformer
from data_loader import load_dataset, load_tokenizer
from utils.trainer import load_optimization, train, test
from utils.predictor import segmentation
from utils.checkpoint import load, save
from utils.config import Config
from utils.distributed import distribute_dataset, setup_device, wrap_model
from utils.metrics import number_of_parameters


def define_argparser():
    parser = argparse.ArgumentParser(description='')

    # loading model
    parser.add_argument('--model', required=True, default='transformer',
                        choices=['transformer', 'cosformer', 'cats', 'graphcodebert'],
                        help='model name')
    parser.add_argument('--model_id', type=str, required=True,
                        help='the unique name of the current model')
    parser.add_argument('--model_name', type=str, default=None,
                        help='checkpoint name to load')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints',
                        help='checkpoint path')
    parser.add_argument('--training', action='store_true', help='if training')
    parser.add_argument('--testing', action='store_true', help='if testing')
    parser.add_argument('--segmentation', type=str, default=None,
                        help='if to segment a file')

    # data_loader
    parser.add_argument('--data_path', type=str, default='./data',
                        help='the path of data files')
    parser.add_argument('--data', required=True, default='binary',
                        choices=['seq', 'binary'], help='datasaets')
    parser.add_argument('--database', type=str, default='./database',
                        help='the path to the packages database')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size of train input data')
    parser.add_argument('--test_batch_size', type=int, default=16,
                        help='batch size of validation & tese input data')
    parser.add_argument('--seq_len', type=int, default=512,
                        help='input sequence length')
    # parser.add_argument('--max_samples', type=int, default=100000,
    #                     help='input sequence length')
    # parser.add_argument('--mem_len', type=int, default=1024,
    #                     help='memory sequence length')

    # hyper-parameters
    transformer.add_args(parser)
    cats.add_args(parser)
    cosformer.add_args(parser)
    graphcodebert.add_args(parser)

    # optimization
    parser.add_argument('--epochs', type=int, default=10,
                        help='train epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='optimizer learning rate')
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
    # parser.add_argument('--patience', type=int, default=3,
    #                     help='early stopping patience')
    # parser.add_argument('--itr', type=int, default=2,
    #                     help='experiments times')

    # distribution
    parser.add_argument('--gpu', action='store_true', help='use gpu or not')
    parser.add_argument('--local_rank', type=int, default=None,
        help="local instance",
    )
    
    return parser


def load_config(arg=None):
    parser = define_argparser()
    config:Config = parser.parse_args(arg)

    # set random seed
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    setup_device(config)
    # load tokenizer
    load_tokenizer(config)
    return config


def load_model(config:Config, output_dim):
    # load model
    models = {
        'transformer': transformer.Transformer,
        'cosformer': cosformer.Cosformer,
        'cats': cats.CATS,
        'graphcodebert': graphcodebert.GraphCodeBERT,
    }

    model = models[config.model](config, output_dim)
    model = wrap_model(config, model)

    if not config.model_name:
        # build name
        config.model_name = '{}_{}_{}_window{}_dim{}'.format(
            config.model_id,
            config.model,
            config.data,
            config.seq_len,
            config.d_model
        )
    return model


def main(arg=None):
    config:Config = load_config(arg)
    
    # load dataset
    train_dataset, val_dataset, test_dataset, output_dim = load_dataset(config)
    train_loader = distribute_dataset(config, train_dataset, config.batch_size)
    val_loader = distribute_dataset(config, val_dataset, config.test_batch_size)
    test_loader = distribute_dataset(config, test_dataset, config.test_batch_size)

    # load model
    model = load_model(config, output_dim)

    # load optimization
    if config.training:
        optimizer, scheduler = load_optimization(config, model)
    else:
        optimizer, scheduler = None, None

    # load checkpoint
    init_epoch = load(config, model, optimizer, scheduler)
    if config.is_host:
        print('Number of parameters for {}: {}' \
          .format(config.model_name, number_of_parameters(model)))

    # training
    if config.training:
        train(config, model, train_loader, val_loader,
            optimizer, scheduler, init_epoch)
        if config.is_host:
            save(config, model, optimizer, scheduler)

    # testing
    if config.testing:
        test(config, model, test_loader)

    # inference
    if config.segmentation:
        segmentation(config, model)

if __name__ == "__main__":
    main()