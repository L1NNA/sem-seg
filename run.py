import argparse
import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim

from utils.config import Config
from baselines.stateful_lstm import StatefulLSTM
from data_loader.js_loader import SimpleSegmentaiton


def train(config:Config, model:nn.Module, train_loader:DataLoader, val_loader:DataLoader):
    model_optim = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        train_loss = []

        model.train()

        for i, (x, y) in enumerate(train_loader):
            model_optim.zero_grad()
            x = x.to(config.device)
            y:torch.Tensor = y.float().to(config.device)
            y = torch.unsqueeze(y, dim = 1)

            outputs, _ = model(x, None)
            loss = criterion(outputs, y)
            train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                print("epoch: {0} | item: {1} | loss: {2:.7f}".format(epoch + 1, i + 1, loss.item()))
            loss.backward()
            model_optim.step()


        # Validation
        with torch.no_grad():
            correct = 0
            total = 0
            for x, y in val_loader:
                x = x.to(config.device)
                y = torch.unsqueeze(y, dim = 1).to(config.device)
                outputs, _ = model(x, None)
                predicted = (outputs>0.5).int()
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        train_loss = np.average(train_loss)
        print("Epoch: {0} | Train Loss: {1:.7f} Accuracy: {2:.2f}%".format(epoch + 1, train_loss, 100 * correct / total))

        # best_model_path = config.weights_path + '/' + 'checkpoint.pth'
        # model.load_state_dict(torch.load(best_model_path))


def main(arg=None):

    parser = argparse.ArgumentParser(description='')

    # basic config
    parser.add_argument('--training', action='store_true', help='status')
    parser.add_argument('--model', required=True, default='stateful_lstm',
                        choices=['cats', 'stateful_lstm'],
                        help='model name, options: [cats, stateful_lstm]')
    parser.add_argument('--model_id', type=str, default='1', help='model id')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # data_loader
    parser.add_argument('--weights_path', type=str, default='./weights', help='the path of weight files')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--seq_len', type=int, default=1024, help='input sequence length')

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
    parser.add_argument('--epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')

    # GPU
    parser.add_argument('--gpu', type=int, action='append')

    args:Config = parser.parse_args(arg)

    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    if torch.cuda.is_available() and args.gpu:
        args.device = args.gpu[0]

    args.model_id = '{}_{}_{}_dm{}_nh{}_el{}_dl{}_df{}_{}'.format(
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

    model = StatefulLSTM(args)
    model = model.to(args.device)
    dataset = SimpleSegmentaiton("C:\\Users\\Leord\\Documents\\GitHub\\sem-seg\\dest")
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [40, 10])
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=True)
    train(args, model, train_loader, val_loader)
    # TODO test

if __name__ == "__main__":
    main([
        "--training",
        "--model_id", "SLSTM",
        "--model", "stateful_lstm",
        "--seq_len", "1024",
        "--e_layers", "2",
        "--des", "'Exp'",
        "--gpu", "0"
    ])