#!/usr/bin/env python3
"""Initializes variables for a Gaussian Mixture Model"""

import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initializes variables for a Gaussian Mixture Model

    Args:
        X (numpy.ndarray): dataset of shape (n, d)
        k (int): number of clusters

    Returns:
        tuple:
            pi (numpy.ndarray): shape (k,) containing priors
            m (numpy.ndarray): shape (k, d) containing means
            S (numpy.ndarray): shape (k, d, d) covariance matrices
        (None, None, None): on failure
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(k, int) or k <= 0 or
            k > X.shape[0]):
        return None, None, None

    n, d = X.shape

    m, _ = kmeans(X, k)

    if m is None:
        return None, None, None

    pi = np.full((k,), 1 / k)

    S = np.tile(np.eye(d), (k, 1, 1))

    return pi, m, S
