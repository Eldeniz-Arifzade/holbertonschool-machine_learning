#!/usr/bin/env python3
"""Module for finding the best number of clusters using BIC."""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Find the best number of clusters for a GMM using BIC.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        kmin: positive integer, minimum number of clusters (inclusive)
        kmax: positive integer, maximum number of clusters (inclusive)
        iterations: positive integer, max iterations for EM algorithm
        tol: non-negative float, tolerance for EM algorithm
        verbose: boolean, whether EM should print information

    Returns:
        best_k, best_result, l, b or None, None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None
    n, d = X.shape
    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax < kmin:
        return None, None, None, None

    k_range = kmax - kmin + 1
    l = np.zeros(k_range)
    b = np.zeros(k_range)
    results = []

    for i, k in enumerate(range(kmin, kmax + 1)):
        pi, m, S, g, log_like = expectation_maximization(
            X, k, iterations, tol, verbose
        )
        if pi is None:
            return None, None, None, None
        # p: number of parameters
        # priors: k-1 (sum to 1)
        # means: k * d
        # covariances: k * d*(d+1)/2 (symmetric)
        p = (k - 1) + k * d + k * d * (d + 1) // 2
        l[i] = log_like
        b[i] = p * np.log(n) - 2 * log_like
        results.append((pi, m, S))

    best_idx = np.argmin(b)
    best_k = kmin + best_idx
    best_result = results[best_idx]

    return best_k, best_result, l, b
