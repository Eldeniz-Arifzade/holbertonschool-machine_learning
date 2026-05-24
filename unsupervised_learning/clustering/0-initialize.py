#!/usr/bin/env python3
"""Initialize cluster centroids for K-means"""

import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means

    Args:
        X (numpy.ndarray): shape (n, d) containing the dataset
        k (int): number of clusters

    Returns:
        numpy.ndarray: shape (k, d) containing initialized centroids
        None: on failure
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(k, int) or k <= 0):
        return None

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    return np.random.uniform(min_vals, max_vals, (k, X.shape[1]))
