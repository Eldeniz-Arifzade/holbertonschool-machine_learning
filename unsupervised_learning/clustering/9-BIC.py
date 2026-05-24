#!/usr/bin/env python3
"""Finds the best number of clusters using BIC"""

import numpy as np

expectation_maximization = __import__(
    '8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000,
        tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using BIC

    Args:
        X (numpy.ndarray): dataset of shape (n, d)
        kmin (int): minimum number of clusters
        kmax (int): maximum number of clusters
        iterations (int): maximum iterations for EM
        tol (float): tolerance for EM
        verbose (bool): verbose mode for EM

    Returns:
        tuple:
            best_k (int): optimal number of clusters
            best_result (tuple): (pi, m, S)
            log_likelihoods (numpy.ndarray): log likelihoods
            bics (numpy.ndarray): BIC values
        (None, None, None, None): on failure
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(kmin, int) or kmin <= 0 or
            not isinstance(iterations, int) or iterations <= 0 or
            not isinstance(tol, float) or tol < 0 or
            not isinstance(verbose, bool)):
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    if (not isinstance(kmax, int) or
            kmax <= 0 or
            kmin >= kmax):
        return None, None, None, None

    size = kmax - kmin + 1

    log_likelihoods = np.zeros(size)
    bics = np.zeros(size)

    best_k = None
    best_result = None

    for i, k in enumerate(range(kmin, kmax + 1)):
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose)

        if pi is None:
            return None, None, None, None

        log_likelihoods[i] = log_likelihood

        p = (k - 1) + (k * d) + (k * d * (d + 1) / 2)

        bics[i] = (p * np.log(n)) - (2 * log_likelihood)

        if best_k is None or bics[i] < bics[best_k - kmin]:
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, log_likelihoods, bics
