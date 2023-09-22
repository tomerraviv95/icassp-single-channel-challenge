import math
import random

import torch
from torch.nn import MSELoss
from torch.optim import Adam

from globals import DEVICE

lr = 1e-3
EPOCHS = 150
BATCH_SIZE = 32


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
        start_index = random.randint(a=0, b=y_data.shape[0] - BATCH_SIZE)
        cur_y_data = torch.Tensor(y_data[start_index:start_index + BATCH_SIZE]).to(DEVICE)
        soft_estimation = net(cur_y_data.unsqueeze(-1))
        cur_x_gt = torch.Tensor(x_gt[start_index:start_index + BATCH_SIZE]).to(DEVICE)
        current_loss = run_train_loop(est=soft_estimation, tx=cur_x_gt, loss_function=loss_function,
                                      optimizer=optimizer)
        loss += current_loss
