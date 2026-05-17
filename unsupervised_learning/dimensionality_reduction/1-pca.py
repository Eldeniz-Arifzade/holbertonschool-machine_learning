#!/usr/bin/env python3
"""PCA v2 module."""

import numpy as np


def pca(X, ndim):
    """Performs PCA on a dataset.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        ndim (int): New dimensionality of the transformed dataset.

    Returns:
        numpy.ndarray: Transformed dataset of shape (n, ndim).
    """
    X_centered = X - np.mean(X, axis=0)

    _, _, Vt = np.linalg.svd(X_centered)

    W = Vt.T[:, :ndim]

    T = np.matmul(X_centered, W)

    return T
