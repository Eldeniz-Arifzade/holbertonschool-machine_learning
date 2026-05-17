#!/usr/bin/env python3
"""PCA module."""

import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) where all features
            are centered.
        var (float): Fraction of variance that must be preserved.

    Returns:
        numpy.ndarray: Weight matrix of shape (d, nd).
    """
    _, S, Vt = np.linalg.svd(X)

    explained_variance = S ** 2
    cumulative_variance = np.cumsum(explained_variance)
    cumulative_variance /= cumulative_variance[-1]

    nd = np.min(np.where(cumulative_variance >= var)) + 1

    W = Vt.T[:, :nd]

    return W
