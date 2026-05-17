#!/usr/bin/env python3
"""P affinities for t-SNE."""

import numpy as np

P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """Calculates the symmetric P affinities of a dataset.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        tol (float): Maximum tolerance allowed for the difference
            in Shannon entropy.
        perplexity (float): Desired perplexity.

    Returns:
        numpy.ndarray: Symmetric P affinity matrix of shape (n, n).
    """
    D, P, betas, H = P_init(X, perplexity)

    n = X.shape[0]

    for i in range(n):
        beta_min = None
        beta_max = None

        Di = np.concatenate((D[i, :i], D[i, i + 1:]))

        Hi, Pi = HP(Di, betas[i])

        H_diff = Hi - H

        while np.abs(H_diff) > tol:
            if H_diff > 0:
                beta_min = betas[i].copy()

                if beta_max is None:
                    betas[i] *= 2
                else:
                    betas[i] = (betas[i] + beta_max) / 2
            else:
                beta_max = betas[i].copy()

                if beta_min is None:
                    betas[i] /= 2
                else:
                    betas[i] = (betas[i] + beta_min) / 2

            Hi, Pi = HP(Di, betas[i])

            H_diff = Hi - H

        P[i, np.concatenate((np.arange(i), np.arange(i + 1, n)))] = Pi

    P = (P + P.T) / (2 * n)

    return P
