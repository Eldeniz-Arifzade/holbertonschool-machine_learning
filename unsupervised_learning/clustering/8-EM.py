#!/usr/bin/env python3
"""Finds the best number of clusters using BIC"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters using BIC.

    Args:
        X (numpy.ndarray): shape (n, d) containing dataset
        kmin (int): minimum number of clusters
        kmax (int): maximum number of clusters
        iterations (int): max iterations for EM
        tol (float): tolerance for EM
        verbose (bool): verbose mode for EM

    Returns:
        tuple: best_k, best_result, l, b or (None, None, None, None) on failure
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(kmin, int) or kmin <= 0 or
            (kmax is not None and
             (not isinstance(kmax, int) or kmax <= 0)) or
            not isinstance(iterations, int) or iterations <= 0 or
            not isinstance(tol, float) or tol < 0 or
            not isinstance(verbose, bool)):
        return None, None, None, None
    n, d = X.shape
    if kmax is None:
        kmax = n
    if kmin >= kmax:
        return None, None, None, None
    l = np.zeros(kmax - kmin + 1)
    b = np.zeros(kmax - kmin + 1)
    best_result = None
    best_k = None
    for i, k in enumerate(range(kmin, kmax + 1)):
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None:
            return None, None, None, None
        l[i] = log_likelihood
        p = (k - 1) + (k * d) + (k * d * (d + 1) // 2)
        b[i] = p * np.log(n) - (2 * log_likelihood)
        if i == 0 or b[i] < np.min(b[:i]):
            best_k = k
            best_result = (pi, m, S)
    return best_k, best_result, l, b
