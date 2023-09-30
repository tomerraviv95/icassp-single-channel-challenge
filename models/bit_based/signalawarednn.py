import math

import torch
import torch.nn as nn

from globals import DEVICE

HIDDEN_SIZE = 200


def binary(x, bits):
    mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


class SignalAwareDNN(nn.Module):
    """
    This implements the VA decoder by a parameterization of the cost calculation by an NN for each stage
    """

    def __init__(self, n_states: int, freq_spacing: int):
        super(SignalAwareDNN, self).__init__()
        self.n_states = n_states
        self.freq_spacing = freq_spacing
        self._initialize_dnn()

    def _initialize_dnn(self):

        layers = [nn.Linear(self.freq_spacing, HIDDEN_SIZE),
                  nn.ReLU(),
                  nn.Linear(HIDDEN_SIZE, self.n_states)]
        self.net = nn.Sequential(*layers).to(DEVICE)

    def forward(self, rx: torch.Tensor, phase: str = 'val') -> torch.Tensor:
        """
        The forward pass of the SignalAwareDNN algorithm
        :param rx: input values, size [batch_size,transmission_length]
        :param phase: 'train' or 'val'
        :returns if in 'train' - the estimated priors [batch_size,transmission_length,n_states]
        if in 'val' - the detected words [n_batch,transmission_length]
        """
        # initialize input probabilities
        reshaped_rx = rx.reshape(rx.shape[0], -1, 32)
        priors = self.net(reshaped_rx)

        if phase != 'training':
            states = torch.argmax(priors, dim=2)
            detected_word = binary(states, bits=math.log2(self.n_states))
            return detected_word
        else:
            return priors
