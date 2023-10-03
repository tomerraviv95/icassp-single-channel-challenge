import math

import torch
import torch.nn as nn

from globals import DEVICE

HIDDEN_SIZE = 32
NUM_LAYERS = 1


def binary(x, bits):
    mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


WINDOW_SIZE = 8


class MixedLSTM(nn.Module):
    """
    This implements the VA decoder by a parameterization of the cost calculation by an NN for each stage
    """

    def __init__(self, n_states: int, freq_spacing: int):
        super(MixedLSTM, self).__init__()
        self.n_states = n_states
        self.freq_spacing = freq_spacing
        self._initialize_dnn()

    def _initialize_dnn(self):
        self.interference_lstm = nn.LSTM(WINDOW_SIZE + self.n_states, HIDDEN_SIZE, NUM_LAYERS, batch_first=True).to(
            DEVICE)
        self.interference_dnn = nn.Sequential(*[nn.ReLU(),
                                                nn.Linear(HIDDEN_SIZE, WINDOW_SIZE)]).to(DEVICE)
        self.normalized_signal_lstm = nn.LSTM(self.freq_spacing, HIDDEN_SIZE, NUM_LAYERS, batch_first=True).to(DEVICE)
        self.normalized_signal_net = nn.Sequential(*[nn.ReLU(),
                                                     nn.Linear(HIDDEN_SIZE, self.n_states)]).to(DEVICE)

    def forward(self, rx: torch.Tensor, phase: str = 'val') -> torch.Tensor:
        """
        The forward pass of the RNN detector
        :param rx: input values, size [batch_size,transmission_length]
        :param phase: 'train' or 'val'
        :return: if in 'train' - the estimated bitwise prob [batch_size,transmission_length,N_CLASSES]
        if in 'val' - the detected words [n_batch,transmission_length]
        """
        # Forward propagate rnn_out: tensor of shape (seq_length, batch_size, input_size)
        reshaped_rx = rx.reshape(rx.shape[0], -1, WINDOW_SIZE)
        states_est_reshaped = torch.zeros(reshaped_rx.shape[0], reshaped_rx.shape[1], 2 * reshaped_rx.shape[2])
        for _ in range(3):
            # Set initial states
            h_n = torch.zeros(NUM_LAYERS, rx.shape[0], HIDDEN_SIZE).to(DEVICE)
            c_n = torch.zeros(NUM_LAYERS, rx.shape[0], HIDDEN_SIZE).to(DEVICE)
            interference_lstm_out, _ = self.interference_lstm(
                torch.cat([reshaped_rx, states_est_reshaped[:, :, :self.n_states]], dim=2),
                (h_n.contiguous(), c_n.contiguous()))
            interference_est = self.interference_dnn(interference_lstm_out)
            # remove the interference estimation from rx
            without_inter_rx = reshaped_rx - interference_est
            reshaped_removed_rx = without_inter_rx.reshape(rx.shape[0], -1, 4 * WINDOW_SIZE)
            # Set initial states
            h_n = torch.zeros(NUM_LAYERS, rx.shape[0], HIDDEN_SIZE).to(DEVICE)
            c_n = torch.zeros(NUM_LAYERS, rx.shape[0], HIDDEN_SIZE).to(DEVICE)
            # Forward propagate rnn_out: tensor of shape (seq_length, batch_size, input_size)
            rnn_out, _ = self.normalized_signal_lstm(reshaped_removed_rx, (h_n.contiguous(), c_n.contiguous()))
            # Linear layer output
            states_est = self.normalized_signal_net(rnn_out)
            states_expanded = states_est.repeat_interleave(repeats=2 * WINDOW_SIZE, dim=1)
            states_est_reshaped = states_expanded.reshape(reshaped_rx.shape[0], reshaped_rx.shape[1], -1)

        if phase != 'training':
            states = torch.argmax(states_est, dim=2)
            detected_word = binary(states, bits=math.log2(self.n_states))
            return detected_word
        else:
            return interference_est, states_est
