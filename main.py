import json
import pickle
import torch
import numpy as np
from torch import nn
import horovod.torch as hvd
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


def train_func():
    hvd.init()

    train_tensor = torch.tensor(np.loadtxt('train.csv', delimiter=','), dtype=torch.float)
    test_tensor = torch.tensor(np.loadtxt('test.csv', delimiter=','), dtype=torch.float)

    train_dataset = TensorDataset(train_tensor[:, :-1], train_tensor[:, -1:])
    test_dataset = TensorDataset(test_tensor[:, :-1], test_tensor[:, -1:])

    ######################################################
    device = f'cuda:{hvd.local_rank()}' if torch.cuda.is_available() else 'cpu'
    # if torch.cuda.is_available():
    #     torch.cuda.set_device(hvd.local_rank())
    torch.set_num_threads(1)

    train_sampler = DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_dl = DataLoader(train_dataset, sampler=train_sampler)

    test_sampler = DistributedSampler(test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_dl = DataLoader(test_dataset, sampler=test_sampler)
    ######################################################

    # define model
    in_dim, hidden_dim, out_dim = 2, 4, 1
    hp = json.load(open('hp.json', 'r'))

    model = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(hp["dropout"]),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(hp["dropout"]),
        nn.Linear(hidden_dim, out_dim)
    )
    model.to(device)

    ######################################################
    opt = Adam(model.parameters(), lr=hp["lr"] * hvd.size())
    opt = hvd.DistributedOptimizer(opt, named_parameters=model.named_parameters(), op=hvd.Average)
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(opt, root_rank=0)
    ######################################################

    loss_fn = MSELoss()
    epoch = 20

    # train and test
    model.train()
    for e in range(epoch):
        ######################################################
        train_sampler.set_epoch(e)
        ######################################################
        avg_loss = []
        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            p = model(x)
            loss = loss_fn(p, y)
            loss.backward()
            opt.step()
            avg_loss.append(loss.item())
        avg_loss = sum(avg_loss) / len(avg_loss)
        print(f'Training. Epoch {e}, MSE loss: {avg_loss}, Worker: {hvd.rank()}')

    model.eval()
    with torch.no_grad():
        avg_loss = []
        for x, y in test_dl:
            x = x.to(device)
            y = y.to(device)
            p = model(x)
            loss = loss_fn(p, y)
            avg_loss.append(loss.item())
        avg_loss = sum(avg_loss) / len(avg_loss)
        ######################################################
        avg_loss = hvd.allreduce(torch.tensor(avg_loss)).item()
        ######################################################
        print(f'Testing. MSE loss: {avg_loss}, Worker: {hvd.rank()}')


if __name__ == '__main__':
    train_func()
