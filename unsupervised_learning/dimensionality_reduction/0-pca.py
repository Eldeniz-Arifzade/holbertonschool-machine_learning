#!/usr/bin/env python3
"""PCA module."""

import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset.

    Args:
        X (numpy.ndarray): Matrix of shape (n, d) containing the dataset.
            All dimensions are centered at 0.
        var (float): Fraction of variance to preserve.

    Returns:
        numpy.ndarray: Weight matrix of shape (d, nd) where nd is the
            number of dimensions needed to maintain the desired variance.
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    explained_variance = (S ** 2)
    cumulative_variance = np.cumsum(explained_variance)
    total_variance = cumulative_variance[-1]

    ratio = cumulative_variance / total_variance
    nd = np.searchsorted(ratio, var) + 1

    W = Vt[:nd].T

    return W
