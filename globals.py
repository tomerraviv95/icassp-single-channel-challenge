import os

import torch
from torch import device

LR = 1e-3
EPOCHS = 100
BATCH_SIZE = 4

DEVICE: device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_FOLDER = os.path.join(DIR_PATH, 'data')
