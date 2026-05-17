#!/usr/bin/env python3
"""Initialize t-SNE variables."""

import numpy as np


def P_init(X, perplexity):
    """Initializes variables for calculating P affinities in t-SNE.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        perplexity (float): Desired perplexity.

    Returns:
        tuple:
            D (numpy.ndarray): Squared pairwise distance matrix of
                shape (n, n).
            P (numpy.ndarray): Initialized P affinity matrix of
                shape (n, n).
            betas (numpy.ndarray): Beta values of shape (n, 1).
            H (float): Shannon entropy corresponding to perplexity.
    """
    sum_X = np.sum(X ** 2, axis=1)

    D = sum_X[:, np.newaxis] + sum_X - 2 * np.matmul(X, X.T)
    np.fill_diagonal(D, 0)

    n = X.shape[0]

    P = np.zeros((n, n))

    betas = np.ones((n, 1))

    H = np.log2(perplexity)

    return D, P, betas, H
