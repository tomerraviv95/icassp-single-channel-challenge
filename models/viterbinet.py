import math

import torch
import torch.nn as nn

from globals import DEVICE

HIDDEN_SIZE = 75


# def create_transition_table(n_states: int) -> np.ndarray:
#     """
#     creates transition table of size [n_states,2]
#     previous state of state i and input bit b is the state in cell [i,b]
#     """
#     transition_table = np.concatenate([np.arange(n_states) for _ in range(n_states)]).reshape(n_states, n_states)
#     return transition_table


# def acs_block(in_prob: torch.Tensor, llrs: torch.Tensor, transition_table: torch.Tensor, n_states: int) -> [
#     torch.Tensor, torch.LongTensor]:
#     """
#     Viterbi ACS block
#     :param in_prob: last stage probabilities, [batch_size,n_states]
#     :param llrs: edge probabilities, [batch_size,1]
#     :param transition_table: transitions
#     :param n_states: number of states
#     :return: current stage probabilities, [batch_size,n_states]
#     """
#     transition_ind = transition_table.reshape(-1).repeat(in_prob.size(0)).long()
#     batches_ind = torch.arange(in_prob.size(0)).repeat_interleave(2 * n_states)
#     trellis = (in_prob + llrs)[batches_ind, transition_ind]
#     reshaped_trellis = trellis.reshape(-1, n_states, 2)
#     return torch.min(reshaped_trellis, dim=2)[0]

def binary(x, bits):
    mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


class ViterbiNet(nn.Module):
    """
    This implements the VA decoder by a parameterization of the cost calculation by an NN for each stage
    """

    def __init__(self, n_states: int, freq_spacing: int):

        super(ViterbiNet, self).__init__()
        self.n_states = n_states
        self.freq_spacing = freq_spacing
        # self.transition_table_array = create_transition_table(n_states)
        # self.transition_table = torch.Tensor(self.transition_table_array).to(DEVICE)
        self._initialize_dnn()

    def _initialize_dnn(self):
        layers = [nn.Linear(self.freq_spacing, HIDDEN_SIZE),
                  nn.ReLU(),
                  nn.Linear(HIDDEN_SIZE, self.n_states)]
        self.net = nn.Sequential(*layers).to(DEVICE)

    def forward(self, rx: torch.Tensor, phase: str = 'val') -> torch.Tensor:
        """
        The forward pass of the ViterbiNet algorithm
        :param rx: input values, size [batch_size,transmission_length]
        :param phase: 'train' or 'val'
        :returns if in 'train' - the estimated priors [batch_size,transmission_length,n_states]
        if in 'val' - the detected words [n_batch,transmission_length]
        """
        # initialize input probabilities
        priors = self.net(rx.reshape(rx.shape[0], -1, 32))

        if phase != 'training':
            # detected_word = torch.zeros(priors.shape[0], priors.shape[1]).to(DEVICE)
            # for i in range(rx.shape[0]):
            #     cur_rx = rx[i].reshape(-1, 32)
            #     for j in range(cur_rx.shape[0]):
            #         # get the lsb of the state
            #         detected_word[i][j] = torch.argmin(in_prob, dim=1) % 2
            #         # run one Viterbi stage
            #         out_prob = acs_block(in_prob, -priors[i][j].reshape(in_prob.shape), self.transition_table, self.n_states)
            #         # update in-probabilities for next layer
            #         in_prob = out_prob
            states = torch.argmax(priors, dim=2)
            detected_word = binary(states, bits=math.log2(self.n_states))
            return detected_word
        else:
            return priors
