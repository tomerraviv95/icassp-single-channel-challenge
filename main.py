import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from data_generation.dataset import generate_datasets, SINR_values
from globals import DEVICE
from models import NetworkType, initialize_networks
from utils.eval import eval_mse
from utils.training import train_networks

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
    train_indices, test_indices = train_test_split(np.array(range(TOTAL_INTERFERENCE_FRAMES)), test_size=TEST_RATIO)
    train_x_gt, train_y_data, train_bits_gt, train_meta_data = generate_datasets(SOI_TYPE, INTERFERENCE_TYPE,
                                                                                 interference_ind=train_indices)
    test_x_gt, test_y_data, test_bits_gt, test_meta_data = generate_datasets(SOI_TYPE, INTERFERENCE_TYPE,
                                                                             interference_ind=test_indices)
    print(f"GT shape:{train_x_gt.shape}")
    print(f"Noisy Data shape:{train_y_data.shape}")
    nets = initialize_networks(model_type)
    train_x_gt,test_x_gt = torch.tensor(train_x_gt,dtype=torch.cfloat).to(DEVICE),torch.tensor(test_x_gt,dtype=torch.cfloat).to(DEVICE)
    train_y_data,test_y_data = torch.tensor(train_y_data,dtype=torch.cfloat).to(DEVICE),torch.tensor(test_y_data,dtype=torch.cfloat).to(DEVICE)
    train_networks(nets, train_x_gt, train_y_data, train_meta_data)
    for sinr,net in zip(SINR_values[-1:],nets[-1:]):
        print(sinr)
        cur_sinr_indices = test_meta_data[:, 1].astype(float) == sinr
        cur_test_x_gt = test_x_gt[cur_sinr_indices]
        cur_test_y_data = test_y_data[cur_sinr_indices]
        cur_real_test_y_data = torch.view_as_real(test_y_data[cur_sinr_indices])
        pred_x_gt = torch.view_as_complex(net(cur_real_test_y_data).reshape(cur_real_test_y_data.shape))
        mse = eval_mse(cur_test_x_gt.detach().numpy(), cur_test_y_data.detach().numpy())
        mse_after_training = eval_mse(cur_test_x_gt.detach().numpy(), pred_x_gt.detach().numpy())
        print(f'Initial MSE: {mse.mean()} -> After training: {mse_after_training.mean()}')
