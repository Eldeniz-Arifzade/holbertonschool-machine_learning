#!/usr/bin/env python3
"""Performs the expectation step in EM for a GMM"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculates the expectation step in the EM algorithm

    Args:
        X (numpy.ndarray): shape (n, d) containing the dataset
        pi (numpy.ndarray): shape (k,) containing cluster priors
        m (numpy.ndarray): shape (k, d) containing cluster means
        S (numpy.ndarray): shape (k, d, d) containing covariance matrices

    Returns:
        tuple:
            g (numpy.ndarray): shape (k, n) posterior probabilities
            l (float): total log likelihood
        (None, None): on failure
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(pi, np.ndarray) or pi.ndim != 1 or
            not isinstance(m, np.ndarray) or m.ndim != 2 or
            not isinstance(S, np.ndarray) or S.ndim != 3):
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if (m.shape != (k, d) or
            S.shape != (k, d, d) or
            not np.isclose(np.sum(pi), 1)):
        return None, None

    try:
        probs = np.array([pi[i] * pdf(X, m[i], S[i])
                          for i in range(k)])

        total = np.sum(probs, axis=0)

        g = probs / total

        likelihood = np.sum(np.log(total))

        return g, likelihood

    except Exception:
        return None, None
