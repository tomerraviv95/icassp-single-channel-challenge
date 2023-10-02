import math
import random

import numpy as np
import tensorflow as tf
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import Adam

from data_generation.dataset import SINR_values
from globals import EPOCHS, BATCH_SIZE, DEVICE, LR
from rfcutils import modulate_qpsk_signal
from utils.eval import eval_ber, eval_mse


def calculate_states(length: int, tx: torch.Tensor) -> torch.Tensor:
    states_enumerator = (2 ** torch.arange(length)).reshape(1, -1).float().to(DEVICE)
    gt_states = torch.sum(tx * states_enumerator, dim=1)
    return gt_states


class MixedWrapper:
    def __init__(self, model, models_num):
        self.nets = [model(4, 32) for _ in range(models_num)]

    def run_train_loop(self, est: torch.Tensor, bits_gt: torch.Tensor, x_gt: torch.Tensor) -> float:
        # calculate loss
        gt_states = calculate_states(2, bits_gt.reshape(-1, 2)).long()
        interference_est, states_est = est
        concat_states_est = torch.concat([states_est[i] for i in range(states_est.shape[0])])
        loss = self.bits_loss_function(input=concat_states_est, target=gt_states)
        loss += self.signal_loss_function(input=interference_est.reshape(x_gt.shape), target=x_gt)
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
            soft_estimation = model(cur_y_data, phase='training')
            cur_x_gt = x_gt[start_index:start_index + BATCH_SIZE]
            cur_bits_gt = bits_gt[start_index:start_index + BATCH_SIZE]
            current_loss = self.run_train_loop(est=soft_estimation, bits_gt=cur_bits_gt, x_gt=cur_x_gt)
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
            real_cur_x_gt = torch.view_as_real(x_gt[current_ind])
            cur_bits_gt = bits_gt[current_ind]
            real_y_data = torch.view_as_real(y_data[current_ind])
            new_model = self.train_single_network(old_model, cur_bits_gt, real_cur_x_gt, real_y_data)
            self.nets[net_ind] = new_model

    def init_loss(self):
        self.bits_loss_function = CrossEntropyLoss().to(DEVICE)
        self.signal_loss_function = MSELoss().to(DEVICE)

    def init_optimizer(self, old_model):
        self.optimizer = Adam(filter(lambda p: p.requires_grad, old_model.parameters()), lr=LR)

    def inference(self, net_ind, cur_test_x_gt, cur_test_y_data, cur_test_bits_gt):
        cur_real_test_y_data = torch.view_as_real(cur_test_y_data)
        pred_bits_gt = self.nets[net_ind](cur_real_test_y_data).reshape(cur_test_bits_gt.shape)
        ber = eval_ber(pred_bits_gt.detach().numpy(), cur_test_bits_gt.detach().numpy())
        avg_log_ber = np.log10(np.mean(ber))
        print(f'log10(BER): {avg_log_ber}[dB]')
        tf_bits = tf.constant(pred_bits_gt.detach().numpy(), dtype=tf.float32)
        pred_x_gt = modulate_qpsk_signal(tf_bits, ebno_db=None)[0]
        mse = eval_mse(cur_test_x_gt.detach().numpy(), cur_test_y_data.detach().numpy())
        mse_after_training = eval_mse(cur_test_x_gt.detach().numpy(), pred_x_gt.numpy())
        print(f'Initial MSE: {mse}[dB] -> After training: {mse_after_training}[dB]')
