import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from data_generation.dataset import generate_datasets
from globals import DEVICE
from models import NETWORKS_TYPES_TO_METHODS, NetworkType
from utils.eval import eval_mse
from utils.training import train_network

SOI_TYPE, INTERFERENCE_TYPE = 'QPSK', 'CommSignal2'

TEST_RATIO = 0.2
SEED = 100
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
TOTAL_INTERFERENCE_FRAMES = 100

if __name__ == "__main__":
    model_type = NetworkType.DNN
    train_indices, test_indices = train_test_split(range(TOTAL_INTERFERENCE_FRAMES), test_size=TEST_RATIO)
    train_x_gt, train_y_data, train_bits_gt, train_meta_data = generate_datasets(SOI_TYPE, INTERFERENCE_TYPE,
                                                                                 interference_ind=train_indices)
    test_x_gt, test_y_data, test_bits_gt, test_meta_data = generate_datasets(SOI_TYPE, INTERFERENCE_TYPE,
                                                                             interference_ind=test_indices)
    print(f"GT shape:{train_x_gt.shape}")
    print(f"Noisy Data shape:{train_y_data.shape}")
    net = NETWORKS_TYPES_TO_METHODS[model_type]
    train_network(net, train_x_gt[train_indices], train_y_data[train_indices])
    mse = eval_mse(test_x_gt[test_indices], test_y_data[test_indices])
    pred_x_gt = net(torch.Tensor(test_y_data[test_indices]).to(DEVICE)).detach().numpy()
    mse_after_training = eval_mse(test_x_gt[test_indices], pred_x_gt)
    for cur_mse, cur_mse_after_training in zip(mse, mse_after_training):
        print(f'Initial MSE: {cur_mse} -> After training: {cur_mse_after_training}')
