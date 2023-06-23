import argparse
import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim

from utils.config import Config
from baselines import transformer, transformer_xl
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

    # loading model
    parser.add_argument('--training', action='store_true', help='status')
    parser.add_argument('--model', required=True, default='transformer',
                        choices=['transformer', 'transformer_xl'],
                        help='model name, options: [TBD]')
    parser.add_argument('--model_name', type=str, required=True, help='the name of weight files')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # data_loader
    parser.add_argument('--data_path', type=str, default='./data', help='the path of data files')
    parser.add_argument('--model', required=True, default='enwik8',
                        choices=['enwik8', 'wikiText103', 'SemSeg'],
                        help='model name, options: [TBD]')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--seq_len', type=int, default=512, help='input sequence length')
    parser.add_argument('--mem_len', type=int, default=0, help='memory sequence length')

    # hyper-parameters
    transformer.add_args(parser)
    transformer_xl.add_args(parser)

    # optimization
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate')
    # parser.add_argument('--patience', type=int, default=3, help='early stopping patience')

    # GPU
    parser.add_argument('--gpu', type=int, action='append')

    args:Config = parser.parse_args(arg)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available() and args.gpu:
        args.device = args.gpu[0]

    model = transformer.Transformer(args)
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