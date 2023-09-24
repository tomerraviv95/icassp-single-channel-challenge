import math
import random

import torch
from torch.nn import MSELoss
from torch.optim import Adam

from data_generation.dataset import SINR_values
from globals import DEVICE

lr = 1e-3
EPOCHS = 100
BATCH_SIZE = 16


def run_train_loop(est: torch.Tensor, tx: torch.Tensor, loss_function, optimizer) -> float:
    # calculate loss
    loss = loss_function(input=est, target=tx)
    current_loss = loss.item()
    # back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return current_loss


def train_networks(nets, x_gt, y_data, train_meta_data):
    loss_function = MSELoss().to(DEVICE)
    for sinr, net in zip(SINR_values[-1:],nets[-1:]):
        print(f"Training for SINR {sinr}")
        optimizer = Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
        current_ind = train_meta_data[:, 1].astype(float) == sinr
        real_x_gt, real_y_data = torch.view_as_real(x_gt[current_ind]), torch.view_as_real(y_data[current_ind])
        train_single_network(loss_function, net, optimizer,real_x_gt,real_y_data)


def train_single_network(loss_function, net, optimizer, x_gt, y_data):
    # run training loops
    loss = 0
    current_loss = math.inf
    for i in range(EPOCHS):
        print(i, current_loss)
        start_index = random.randint(a=0, b=y_data.shape[0] - BATCH_SIZE)
        cur_y_data = y_data[start_index:start_index + BATCH_SIZE]
        soft_estimation = net(cur_y_data)
        cur_x_gt = x_gt[start_index:start_index + BATCH_SIZE]
        cur_x_gt = cur_x_gt.reshape(soft_estimation.shape)
        current_loss = run_train_loop(est=soft_estimation, tx=cur_x_gt, loss_function=loss_function,
                                      optimizer=optimizer)
        loss += current_loss
