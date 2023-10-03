import numpy as np

comm2_signal = np.load('outputs/TomerSubmission_TestSet1Mixture_estimated_soi_QPSK_CommSignal2.npy')
comm3_signal = np.load('outputs/TomerSubmission_TestSet1Mixture_estimated_soi_QPSK_CommSignal3.npy')
print(comm2_signal.shape)
print(np.sum(comm2_signal == comm3_signal) / (comm2_signal.shape[0] * comm2_signal.shape[1]))

comm2_bits = np.load('outputs/TomerSubmission_TestSet1Mixture_estimated_bits_QPSK_CommSignal2.npy')
comm3_bits = np.load('outputs/TomerSubmission_TestSet1Mixture_estimated_bits_QPSK_CommSignal3.npy')
print(comm2_bits.shape)
print(np.sum(comm2_bits == comm3_bits) / (comm2_bits.shape[0] * comm2_bits.shape[1]))
