#!/usr/bin/env python3
"""Performs the maximization step in EM for a GMM"""

import numpy as np


def maximization(X, g):
    """Calculates the maximization step in the EM algorithm

    Args:
        X (numpy.ndarray): shape (n, d) containing the dataset
        g (numpy.ndarray): shape (k, n) containing posterior probabilities

    Returns:
        tuple:
            pi (numpy.ndarray): shape (k,) updated priors
            m (numpy.ndarray): shape (k, d) updated means
            S (numpy.ndarray): shape (k, d, d) updated covariance matrices
        (None, None, None): on failure
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(g, np.ndarray) or g.ndim != 2):
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape

    if n != n_g or not np.isclose(np.sum(g, axis=0), 1).all():
        return None, None, None

    try:
        weights = np.sum(g, axis=1)

        pi = weights / n

        m = (g @ X) / weights[:, np.newaxis]

        S = np.zeros((k, d, d))

        for i in range(k):
            diff = X - m[i]
            weighted_diff = g[i][:, np.newaxis] * diff
            S[i] = (weighted_diff.T @ diff) / weights[i]

        return pi, m, S

    except Exception:
        return None, None, None
