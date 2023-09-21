import torch
from torch import device

DEVICE: device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_FOLDER: str = 'data'
