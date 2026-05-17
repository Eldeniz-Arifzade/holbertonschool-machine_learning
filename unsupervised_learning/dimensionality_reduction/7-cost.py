#!/usr/bin/env python3
"""Cost calculation for t-SNE."""

import numpy as np


def cost(P, Q):
    """Calculates the cost of the t-SNE transformation.

    Args:
        P (numpy.ndarray): P affinities of shape (n, n).
        Q (numpy.ndarray): Q affinities of shape (n, n).

    Returns:
        float: Cost of the transformation.
    """
    P = np.maximum(P, 1e-12)
    Q = np.maximum(Q, 1e-12)

    C = np.sum(P * np.log(P / Q))

    return C
