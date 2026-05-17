#!/usr/bin/env python3
"""Gradient calculation for t-SNE."""

import numpy as np

Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """Calculates the gradients of Y.

    Args:
        Y (numpy.ndarray): Low-dimensional representation of shape
            (n, ndim).
        P (numpy.ndarray): P affinities of shape (n, n).

    Returns:
        tuple:
            dY (numpy.ndarray): Gradients of shape (n, ndim).
            Q (numpy.ndarray): Q affinities of shape (n, n).
    """
    Q, num = Q_affinities(Y)

    PQ = (P - Q) * num

    dY = np.zeros_like(Y)

    for i in range(Y.shape[0]):
        diff = Y[i] - Y
        dY[i] = np.sum(PQ[:, i][:, np.newaxis] * diff, axis=0)

    return dY, Q
