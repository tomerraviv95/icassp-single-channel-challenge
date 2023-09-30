import random
from enum import Enum

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from data_generation.dataset import generate_datasets, SINR_values
from globals import DEVICE
from models import WrapperType, TYPES_TO_WRAPPER, NetworkType, NETWORKS_TYPES_TO_METHODS

SOI_TYPE = 'QPSK'


class INTERFERENCE_TYPE(Enum):
    CommSignal2 = 'CommSignal2'
    CommSignal3 = 'CommSignal3'
    CommSignal5G1 = 'CommSignal5G1'
    EMISignal1 = 'EMISignal1'


TEST_RATIO = 0.2
SEED = 100
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
TOTAL_INTERFERENCE_FRAMES = 100

if __name__ == "__main__":
    model_wrapper = WrapperType.ModelFree
    model_type = NetworkType.WAVE
    interference = INTERFERENCE_TYPE.CommSignal3.name
    ######################
    model = NETWORKS_TYPES_TO_METHODS[model_type]
    wrapper = TYPES_TO_WRAPPER[model_wrapper](model, len(SINR_values))
    train_indices, test_indices = train_test_split(np.array(range(TOTAL_INTERFERENCE_FRAMES)), test_size=TEST_RATIO)
    train_x_gt, train_y_data, train_bits_gt, train_meta_data = generate_datasets(SOI_TYPE, interference,
                                                                                 interference_ind=train_indices)
    test_x_gt, test_y_data, test_bits_gt, test_meta_data = generate_datasets(SOI_TYPE, interference,
                                                                             interference_ind=train_indices)
    print(f"GT shape:{train_x_gt.shape}")
    print(f"Noisy Data shape:{train_y_data.shape}")
    train_bits_gt, test_bits_gt = torch.tensor(train_bits_gt).to(DEVICE), torch.tensor(test_bits_gt).to(DEVICE)
    train_x_gt, test_x_gt = torch.tensor(train_x_gt, dtype=torch.cfloat).to(DEVICE), \
                            torch.tensor(test_x_gt, dtype=torch.cfloat).to(DEVICE)
    train_y_data, test_y_data = torch.tensor(train_y_data, dtype=torch.cfloat).to(DEVICE), \
                                torch.tensor(test_y_data, dtype=torch.cfloat).to(DEVICE)
    wrapper.train_networks(train_x_gt, train_y_data, train_bits_gt, train_meta_data)
    indices = range(len(SINR_values))
    for net_ind in indices:
        sinr = SINR_values[net_ind]
        print(sinr)
        cur_sinr_indices = test_meta_data[:, 1].astype(float) == sinr
        cur_test_x_gt = test_x_gt[cur_sinr_indices]
        cur_test_y_data = test_y_data[cur_sinr_indices]
        cur_test_bits_gt = test_bits_gt[cur_sinr_indices]
        wrapper.inference(net_ind, cur_test_x_gt, cur_test_y_data, cur_test_bits_gt)
