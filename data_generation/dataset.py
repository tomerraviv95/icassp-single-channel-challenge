import numpy as np
import tensorflow as tf

from data_generation import SIGNAL_LENGTH
from data_generation.interference import load_interference
from data_generation.signal import get_soi_generation_fn

FRAMES_NUM = 100
SINR_values = np.arange(-30, 0.1, 3)
get_db = lambda p: 10 * np.log10(p)
get_pow = lambda s: np.mean(np.abs(s) ** 2, axis=-1)
get_sinr = lambda s, i: get_pow(s) / get_pow(i)
get_sinr_db = lambda s, i: get_db(get_sinr(s, i))


def generate_datasets(soi_type, interference_sig_type, interference_ind):
    generate_soi, demod_soi = get_soi_generation_fn(soi_type)
    sig_data, sig_type_info = load_interference(interference_sig_type)

    y_data, x_gt, bits_gt, meta_data = [], [], [], []
    for idx, sinr in enumerate(SINR_values):
        sig1, _, bits1, _ = generate_soi(FRAMES_NUM, SIGNAL_LENGTH)
        sig2 = sig_data[np.random.choice(interference_ind, size=(FRAMES_NUM)), :]

        sig_target = sig1[:, :SIGNAL_LENGTH]

        rand_start_idx2 = np.random.randint(sig2.shape[1] - SIGNAL_LENGTH, size=sig2.shape[0])
        inds2 = tf.cast(rand_start_idx2.reshape(-1, 1) + np.arange(SIGNAL_LENGTH).reshape(1, -1), tf.int32)
        sig_interference = tf.experimental.numpy.take_along_axis(sig2, inds2, axis=1)

        # Interference Coefficient
        rand_gain = np.sqrt(10 ** (-sinr / 10)).astype(np.float32)
        rand_phase = tf.random.uniform(shape=[sig_interference.shape[0], 1])
        rand_gain = tf.complex(rand_gain, tf.zeros_like(rand_gain))
        rand_phase = tf.complex(rand_phase, tf.zeros_like(rand_phase))
        coeff = rand_gain * tf.math.exp(1j * 2 * np.pi * rand_phase)

        sig_mixture = sig_target + sig_interference * coeff

        y_data.append(sig_mixture)
        x_gt.append(sig_target)
        bits_gt.append(bits1)

        actual_sinr = get_sinr_db(sig_target, sig_interference * coeff)
        meta_data.append(np.vstack(([rand_gain.numpy().real for _ in range(FRAMES_NUM)],
                                    [sinr for _ in range(FRAMES_NUM)], actual_sinr,
                                    [soi_type for _ in range(FRAMES_NUM)],
                                    [interference_sig_type for _ in range(FRAMES_NUM)])))

    with tf.device('CPU'):
        y_data = tf.concat(y_data, axis=0).numpy()
        x_gt = tf.concat(x_gt, axis=0).numpy()
        bits_gt = tf.concat(bits_gt, axis=0).numpy()

    meta_data = np.concatenate(meta_data, axis=1).T
    return x_gt, y_data, bits_gt, meta_data
