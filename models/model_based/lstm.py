import math

import torch
import torch.nn as nn

from globals import DEVICE

HIDDEN_SIZE = 100
NUM_LAYERS = 1


def binary(x, bits):
    mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


class SignalAwareLSTM(nn.Module):
    """
    This implements the VA decoder by a parameterization of the cost calculation by an NN for each stage
    """

    def __init__(self, n_states: int, freq_spacing: int):
        super(SignalAwareLSTM, self).__init__()
        self.n_states = n_states
        self.freq_spacing = freq_spacing
        self._initialize_dnn()

    def _initialize_dnn(self):
        self.lstm = nn.LSTM(self.freq_spacing, HIDDEN_SIZE, NUM_LAYERS, batch_first=True).to(DEVICE)
        layers = [nn.ReLU(),
                  nn.Linear(HIDDEN_SIZE, self.n_states)]
        self.net = nn.Sequential(*layers).to(DEVICE)

    def forward(self, rx: torch.Tensor, phase: str = 'val') -> torch.Tensor:
        """
        The forward pass of the RNN detector
        :param rx: input values, size [batch_size,transmission_length]
        :param phase: 'train' or 'val'
        :return: if in 'train' - the estimated bitwise prob [batch_size,transmission_length,N_CLASSES]
        if in 'val' - the detected words [n_batch,transmission_length]
        """
        # Set initial states
        h_n = torch.zeros(NUM_LAYERS, rx.shape[0], HIDDEN_SIZE).to(DEVICE)
        c_n = torch.zeros(NUM_LAYERS, rx.shape[0], HIDDEN_SIZE).to(DEVICE)
        reshaped_rx = rx.reshape(rx.shape[0], -1, 32)

        # Forward propagate rnn_out: tensor of shape (seq_length, batch_size, input_size)
        rnn_out, _ = self.lstm(reshaped_rx, (h_n.contiguous(), c_n.contiguous()))

        # Linear layer output
        out = self.net(rnn_out)

        if phase != 'training':
            states = torch.argmax(out, dim=2)
            detected_word = binary(states, bits=math.log2(self.n_states))
            return detected_word
        else:
            return out
