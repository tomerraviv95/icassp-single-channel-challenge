import os
import random
from enum import Enum

import numpy as np
import tensorflow as tf
import torch

from data_generation.dataset import generate_datasets, SINR_values
from globals import DEVICE, DATA_FOLDER
from models import WrapperType, TYPES_TO_WRAPPER, NetworkType, NETWORKS_TYPES_TO_METHODS

SOI_TYPE = 'QPSK'
TESTSET_IDENTIFIER = 'TestSet1Mixture'


class INTERFERENCE_TYPE(Enum):
    CommSignal2 = 'CommSignal2'
    CommSignal3 = 'CommSignal3'
    CommSignal5G1 = 'CommSignal5G1'
    EMISignal1 = 'EMISignal1'


ID = "TomerSubmission"

SEED = 100
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
TOTAL_INTERFERENCE_FRAMES = 100

if __name__ == "__main__":
    model_wrapper = WrapperType.Mixed
    model_type = NetworkType.MixedLSTM
    for interference in [INTERFERENCE_TYPE.CommSignal3.name,
                         INTERFERENCE_TYPE.CommSignal2.name,
                         INTERFERENCE_TYPE.CommSignal5G1.name,
                         INTERFERENCE_TYPE.EMISignal1.name]:
        print(interference)
        model = NETWORKS_TYPES_TO_METHODS[model_type]
        wrapper = TYPES_TO_WRAPPER[model_wrapper](model, len(SINR_values))
        train_indices = np.array(range(TOTAL_INTERFERENCE_FRAMES))
        train_x_gt, train_y_data, train_bits_gt, train_meta_data, train_sig_interferences = generate_datasets(SOI_TYPE,
                                                                                                              interference,
                                                                                                              interference_ind=train_indices)
        print(f"GT shape:{train_x_gt.shape}")
        print(f"Noisy Data shape:{train_y_data.shape}")
        train_bits_gt = torch.tensor(train_bits_gt).to(DEVICE)
        train_x_gt = torch.tensor(train_x_gt, dtype=torch.cfloat).to(DEVICE)
        train_y_data = torch.tensor(train_y_data, dtype=torch.cfloat).to(DEVICE)
        wrapper.train_networks(train_x_gt, train_y_data, train_bits_gt, train_meta_data)
        test_y_data = np.load(
            os.path.join(DATA_FOLDER, 'test', f'{TESTSET_IDENTIFIER}_testmixture_{SOI_TYPE}_{interference}.npy'))
        test_meta_data = np.load(
            os.path.join(DATA_FOLDER, 'test', f'{TESTSET_IDENTIFIER}_testmixture_{SOI_TYPE}_{interference}_metadata.npy'))
        indices = range(len(SINR_values))
        total_pred_x_gt, total_pred_bits_gt = [], []
        for net_ind in indices:
            sinr = SINR_values[net_ind]
            print(sinr)
            cur_sinr_indices = test_meta_data[:, 1].astype(float) == sinr
            cur_test_y_data = torch.tensor(test_y_data[cur_sinr_indices], dtype=torch.cfloat).to(DEVICE)
            pred_x_gt, pred_bits_gt = wrapper.forward(net_ind, cur_test_y_data)
            total_pred_x_gt.append(pred_x_gt)
            total_pred_bits_gt.append(pred_bits_gt)

        sig_est = tf.concat(total_pred_x_gt, axis=0).numpy()
        bit_est = tf.concat(total_pred_bits_gt, axis=0).numpy()
        np.save(
            os.path.join('outputs', f'{ID}_{TESTSET_IDENTIFIER}_estimated_soi_{SOI_TYPE}_{interference}'),
            sig_est)
        np.save(
            os.path.join('outputs', f'{ID}_{TESTSET_IDENTIFIER}_estimated_bits_{SOI_TYPE}_{interference}'),
            bit_est)
