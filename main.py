import math
import os
import pickle
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam

HIDDEN_SIZE = 256
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_pkl(filename):
    with open(f'{filename}.pickle', 'rb') as handle:
        data = pickle.load(handle)
    return data


class DNNDetector(nn.Module):
    """
    The DNNDetector Network Architecture
    """

    def __init__(self):
        super(DNNDetector, self).__init__()
        layers = [nn.Linear(TOTAL_INDEX, HIDDEN_SIZE),
                  nn.ReLU(),
                  nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                  nn.ReLU(),
                  nn.Linear(HIDDEN_SIZE, TOTAL_INDEX)]
        self.net = nn.Sequential(*layers).to(DEVICE)

    def forward(self, rx: torch.Tensor) -> torch.Tensor:
        soft_estimation = self.net(rx)
        return soft_estimation


lr = 1e-3
EPOCHS = 150
TOTAL_INDEX = 40960


def run_train_loop(est: torch.Tensor, tx: torch.Tensor, loss_function, optimizer) -> float:
    # calculate loss
    loss = loss_function(input=est, target=tx)
    current_loss = loss.item()
    # back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return current_loss


def train_network(net, x_gt, y_data):
    optimizer = Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    loss_function = MSELoss().to(DEVICE)
    # run training loops
    loss = 0
    current_loss = math.inf
    for i in range(EPOCHS):
        print(i, current_loss)
        # pass through detector
        # start_index = random.randint(a=0, b=1100 - 64)
        start_index = 0
        cur_y_data = torch.Tensor(y_data[start_index:start_index + 1, :TOTAL_INDEX]).to(DEVICE)
        soft_estimation = net(cur_y_data)
        cur_x_gt = torch.Tensor(x_gt[start_index:start_index + 1, :TOTAL_INDEX]).to(DEVICE)
        current_loss = run_train_loop(est=soft_estimation, tx=cur_x_gt, loss_function=loss_function,
                                      optimizer=optimizer)
        loss += current_loss


def eval_mse(x_test, y_test):
    assert x_test.shape == y_test.shape, 'Invalid SOI estimate shape'
    return np.mean(np.abs(x_test - y_test) ** 2, axis=1)


DATA_FOLDER = 'data'

SEED = 100
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

if __name__ == "__main__":
    x_gt = load_pkl(os.path.join(DATA_FOLDER, 'all_sig1'))
    print(f"GT shape:{x_gt.shape}")
    y_data = load_pkl(os.path.join(DATA_FOLDER, 'all_sig_mixture'))
    print(f"Noisy Data shape:{y_data.shape}")
    meta = load_pkl(os.path.join(DATA_FOLDER, 'meta_data'))
    indices = range(x_gt.shape[0])
    train_indices, test_indices, train_meta, test_meta = train_test_split(indices, meta[:, 1], stratify=meta[:, 1],
                                                                          test_size=0.2)
    net = DNNDetector()
    train_network(net, x_gt[train_indices], y_data[train_indices])
    mse = eval_mse(x_gt[test_indices, :TOTAL_INDEX], y_data[test_indices, :TOTAL_INDEX])
    mse_after_training = eval_mse(x_gt[test_indices, :TOTAL_INDEX],
                                  net(torch.Tensor(y_data[test_indices, :TOTAL_INDEX]).to(DEVICE)).detach().numpy())
    for cur_mse, cur_mse_after_training in zip(mse, mse_after_training):
        print(f'Initial MSE: {cur_mse} -> After training: {cur_mse_after_training}')
