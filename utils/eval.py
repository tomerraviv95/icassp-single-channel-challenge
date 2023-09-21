import numpy as np


def eval_mse(x_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """

    :rtype: object
    """
    assert x_test.shape == y_test.shape, 'Invalid SOI estimate shape'
    return 10 * np.log10(np.mean(np.abs(x_test - y_test) ** 2, axis=1))
