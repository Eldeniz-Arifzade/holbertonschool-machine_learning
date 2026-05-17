#!/usr/bin/env python3
"""Q affinities for t-SNE."""

import numpy as np


def Q_affinities(Y):
    """Calculates the Q affinities.

    Args:
        Y (numpy.ndarray): Low-dimensional representation of shape
            (n, ndim).

    Returns:
        tuple:
            Q (numpy.ndarray): Q affinities of shape (n, n).
            num (numpy.ndarray): Numerators of shape (n, n).
    """
    sum_Y = np.sum(Y ** 2, axis=1)

    D = sum_Y[:, np.newaxis] + sum_Y - 2 * np.matmul(Y, Y.T)

    num = 1 / (1 + D)

    np.fill_diagonal(num, 0)

    Q = num / np.sum(num)

    return Q, num
