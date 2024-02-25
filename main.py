import argparse
import random
from os.path import join

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from models import cats, cosformer, graphcodebert, transformer, multi_cats
from models import chain_of_experts, autobert, longformer
from utils.data_utils import load_dataset, load_tokenizer, DATASET_MAP
from utils.trainer import load_optimization, train, test, train_siamese, test_siamese, train_coe, test_coe
from utils.checkpoint import load, save
from utils.config import Config
from utils.distributed import distribute_dataset, setup_device, distribute_model
from utils.metrics import number_of_parameters


models = {
    'transformer': transformer.Transformer,
    'cosformer': cosformer.Cosformer,
    'cats': cats.CATS,
    'graphcodebert': graphcodebert.GraphCodeBERT,
    'autobert': autobert.AutoBERT,
    'coe': chain_of_experts.ChainOfExperts,
    'multi_cats': multi_cats.MultiCATS,
    'longformer': longformer.Longformer
}


def define_argparser():
    parser = argparse.ArgumentParser(description='')

    # loading model
    parser.add_argument('--model', required=True,
                        choices=models.keys(),
                        help='model name')
    parser.add_argument('--model_id', type=str, required=True,
                        help='the unique name of the current model')
    parser.add_argument('--model_name', type=str, default=None,
                        help='checkpoint name to load')
    parser.add_argument('--comment', type=str, default='',
                        help='comment')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints',
                        help='checkpoint path')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='log path')
    parser.add_argument('--training', action='store_true', help='if training')
    parser.add_argument('--validation', action='store_true', help='if validation')
    parser.add_argument('--testing', action='store_true', help='if testing')
    # parser.add_argument('--segmentation', type=str, default=None,
    #                     help='if to segment a file')

    # data_loader
    parser.add_argument('--data_path', type=str, default='./data',
                        help='the path of data files')
    parser.add_argument('--data', required=True, help='datasaets',
                        choices=DATASET_MAP.keys())
    # parser.add_argument('--database', type=str, default='./database',
    #                     help='the path to the packages database')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--seq_len', type=int, default=512,
                        help='input sequence length')
    parser.add_argument('--skip_label', type=int, default=0,
                        help='number of labels to')                    

    # hyper-parameters
    transformer.add_args(parser)
    cats.add_args(parser)
    cosformer.add_args(parser)
    graphcodebert.add_args(parser)
    autobert.add_args(parser)
    chain_of_experts.add_args(parser)

    # optimization
    parser.add_argument('--epochs', type=int, default=10,
                        help='train epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='optimizer learning rate')
    parser.add_argument("--optim", type=str, default="adam",
                        choices=("sgd", "adam"),
                        help="optimization method",
    )

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
    model = models[config.model](config, output_dim)

    if not config.model_name:
        # build name
        config.model_name = '{}_{}_{}_window{}_dim{}{}'.format(
            config.model_id,
            config.model,
            config.data,
            config.seq_len,
            config.d_model,
            ('_' + config.comment) if config.comment else ''
        )
    return model


def main(arg=None):
    config:Config = load_config(arg)
    
    # load dataset
    train_dataset, val_dataset, test_dataset, output_dim = load_dataset(config)
    train_loader = distribute_dataset(config, train_dataset, config.batch_size)
    val_loader = distribute_dataset(config, val_dataset, config.batch_size)
    test_loader = distribute_dataset(config, test_dataset, config.batch_size)

    trainer = train
    tester = test
    if config.data == 'siamese_clone':
        trainer, tester = train_siamese, test_siamese
    elif config.data == 'coe':
        trainer, tester = train_coe, test_coe

    # load model
    model = load_model(config, output_dim)
    writer = SummaryWriter(join(config.log_dir, config.model_name)) if config.is_host else None

    # load optimization
    if config.training:
        optimizer = load_optimization(config, model)
    else:
        optimizer = None

    # distribute model after weights loaded
    model = distribute_model(config, model)
    # load checkpoint
    init_epoch = load(config, model, optimizer)
    if config.is_host:
        print('Number of parameters for {}: {}' \
          .format(config.model_name, number_of_parameters(model)))
    

    # training
    if config.training:
        trainer(config, model, train_loader, val_loader,
            optimizer, writer, init_epoch)
        if config.is_host:
            save(config, model, optimizer)
    elif config.validation:
        tester(config, model, val_loader, 'Validation', writer, init_epoch)
    

    # testing
    if config.testing:
        tester(config, model, test_loader, 'Test', writer, init_epoch)

if __name__ == "__main__":
    main()