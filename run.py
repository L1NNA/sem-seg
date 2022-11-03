import argparse
import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim


from utils.config import Config


def train(config:Config, model:nn.Module, data_loader:DataLoader):
    model_optim = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(config.training_epochs):
        iter_count = 0
        train_loss = []

        model.train()

        # state
        state = torch.zeros(x.size()).float()
        for i, (x, y) in enumerate(data_loader):
            iter_count += 1
            model_optim.zero_grad()
            x = x.float().to(config.device)
            y = y.float().to(config.device)

            outputs, state = model(x, state)
            loss = criterion(outputs, y)
            train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
            loss.backward()
            model_optim.step()


        train_loss = np.average(train_loss)
        print("Epoch: {0} | Train Loss: {2:.7f}".format(epoch + 1, train_loss))

        best_model_path = config.weights_path + '/' + 'checkpoint.pth'
        model.load_state_dict(torch.load(best_model_path))


def main():

    parser = argparse.ArgumentParser(description='')

    # basic config
    parser.add_argument('--training', action='store_true', help='status')
    parser.add_argument('--model', required=True, default='stateful_transformer',
                        choices=['stateful_transformer', 'cats', 'stateful_lstm'],
                        help='model name, options: [stateful_transformer, cats, stateful_lstm]')
    parser.add_argument('--model_id', type=str, default='1', help='model id')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # data_loader
    parser.add_argument('--weights_path', type=str, default='./weights', help='the path of weight files')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')

    # hyper-parameters
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')

    # GPU
    parser.add_argument('--gpu', type=int, action='append')

    args:Config = parser.parse_args()

    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    if torch.cuda.is_available() and args.gpu:
        args.device = args.gpu[0]

    model_id = '{}_{}_{}_dm{}_nh{}_el{}_dl{}_df{}_{}'.format(
        args.model_id,
        args.model,
        args.seq_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.des
    )

    


if __name__ == "__main__":
    main()