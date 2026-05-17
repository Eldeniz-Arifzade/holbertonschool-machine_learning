#!/usr/bin/env python3
"""Module for performing Principal Component Analysis (PCA)."""
import numpy as np


def pca(X, var=0.95):
    """Perform PCA on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) where n is the number of data points
           and d is the number of dimensions. All dimensions have a mean of 0.
        var: fraction of the variance that the PCA transformation should
             maintain (default 0.95).

    Returns:
        W: numpy.ndarray of shape (d, nd) — the weights matrix that maintains
           var fraction of X's original variance, where nd is the new
           dimensionality of the transformed X.
    """
    _, s, Vt = np.linalg.svd(X)
    explained = np.cumsum(s ** 2) / np.sum(s ** 2)
    nd = np.argmax(explained >= var) + 1
    return Vt[:nd].T
