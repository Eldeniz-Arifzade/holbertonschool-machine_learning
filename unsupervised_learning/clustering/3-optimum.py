#!/usr/bin/env python3
"""Determines the optimum number of clusters"""

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Tests for the optimum number of clusters by variance

    Args:
        X (numpy.ndarray): dataset of shape (n, d)
        kmin (int): minimum number of clusters
        kmax (int): maximum number of clusters
        iterations (int): maximum iterations for K-means

    Returns:
        tuple:
            results (list): [(C, clss), ...] for each k
            d_vars (list): variance differences
        (None, None): on failure
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(kmin, int) or kmin <= 0 or
            kmax is None or
            not isinstance(kmax, int) or kmax <= 0 or
            kmax <= kmin or
            not isinstance(iterations, int) or iterations <= 0):
        return None, None

    results = []
    variances = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)

        if C is None or clss is None:
            return None, None

        results.append((C, clss))
        variances.append(variance(X, C))

    base_var = variances[0]
    d_vars = [base_var - v for v in variances]

    return results, d_vars
