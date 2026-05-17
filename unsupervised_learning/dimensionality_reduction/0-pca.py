#!/usr/bin/env python3
"""PCA module."""

import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) with centered features.
        var (float): Fraction of variance to preserve.

    Returns:
        numpy.ndarray: Weight matrix of shape (d, nd).
    """
    _, S, Vt = np.linalg.svd(X)

    explained_variance = S ** 2
    ratio = np.cumsum(explained_variance) / np.sum(explained_variance)

    nd = np.where(ratio > var)[0][0] + 1

    return Vt.T[:, :nd]
