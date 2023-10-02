import math
import random

import torch
from torch.nn import MSELoss
from torch.optim import Adam

from data_generation.dataset import SINR_values
from globals import DEVICE, BATCH_SIZE, LR, EPOCHS
from utils.eval import eval_mse


class SignalBasedWrapper:
    def __init__(self, model, models_num):
        self.nets = [model() for _ in range(models_num)]

    def run_train_loop(self, est: torch.Tensor, tx: torch.Tensor) -> float:
        # calculate loss
        loss = self.loss_function(input=est, target=tx)
        current_loss = loss.item()
        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return current_loss

    def train_single_network(self, model, bits_gt, x_gt, y_data):
        # run in_training loops
        loss = 0
        current_loss = math.inf
        for i in range(EPOCHS):
            print(i, current_loss)
            start_index = random.randint(a=0, b=y_data.shape[0] - BATCH_SIZE)
            cur_y_data = y_data[start_index:start_index + BATCH_SIZE]
            soft_estimation = model(cur_y_data)
            cur_x_gt = x_gt[start_index:start_index + BATCH_SIZE]
            cur_x_gt = cur_x_gt.reshape(soft_estimation.shape)
            current_loss = self.run_train_loop(est=soft_estimation, tx=cur_x_gt)
            loss += current_loss
        return model

    def train_networks(self, x_gt, y_data, bits_gt, meta_data):
        self.init_loss()
        indices = range(len(SINR_values))
        for net_ind in indices:
            sinr = SINR_values[net_ind]
            print(f"Training for SINR {sinr}")
            old_model = self.nets[net_ind]
            self.init_optimizer(old_model)
            current_ind = meta_data[:, 1].astype(float) == sinr
            real_x_gt = torch.view_as_real(x_gt[current_ind])
            real_y_data = torch.view_as_real(y_data[current_ind])
            new_model = self.train_single_network(old_model, bits_gt, real_x_gt, real_y_data)
            self.nets[net_ind] = new_model

    def init_loss(self):
        self.loss_function = MSELoss().to(DEVICE)

    def init_optimizer(self, old_model):
        self.optimizer = Adam(filter(lambda p: p.requires_grad, old_model.parameters()), lr=LR)

    def forward(self, net_ind, cur_test_y_data):
        cur_real_test_y_data = torch.view_as_real(cur_test_y_data)
        pred_x_gt = torch.view_as_complex(self.nets[net_ind](cur_real_test_y_data).reshape(cur_real_test_y_data.shape))
        return pred_x_gt

    def inference(self, net_ind, cur_test_x_gt, cur_test_y_data, cur_test_bits_gt):
        pred_x_gt = self.forward(net_ind, cur_test_y_data)
        mse = eval_mse(cur_test_x_gt.detach().numpy(), cur_test_y_data.detach().numpy())
        mse_after_training = eval_mse(cur_test_x_gt.detach().numpy(), pred_x_gt.detach().numpy())
        print(f'Initial MSE: {mse}[dB] -> After training: {mse_after_training}[dB]')
