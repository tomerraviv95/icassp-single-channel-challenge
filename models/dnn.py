import torch
from torch import nn

from data_generation import SIGNAL_LENGTH
from globals import DEVICE

HIDDEN_SIZE = 256


class DNNDetector(nn.Module):
    """
    The DNNDetector Network Architecture
    """

    def __init__(self):
        super(DNNDetector, self).__init__()
        layers = [nn.Linear(SIGNAL_LENGTH, HIDDEN_SIZE),
                  nn.ReLU(),
                  nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                  nn.ReLU(),
                  nn.Linear(HIDDEN_SIZE, SIGNAL_LENGTH)]
        self.net = nn.Sequential(*layers).to(DEVICE)

    def forward(self, rx: torch.Tensor) -> torch.Tensor:
        soft_estimation = self.net(rx)
        return soft_estimation
