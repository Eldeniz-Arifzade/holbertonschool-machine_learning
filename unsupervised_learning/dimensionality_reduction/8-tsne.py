#!/usr/bin/env python3
"""t-SNE implementation."""

import numpy as np

pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0,
         iterations=1000, lr=500):
    """Performs a t-SNE transformation.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        ndims (int): Output dimensionality.
        idims (int): Intermediate PCA dimensionality.
        perplexity (float): Perplexity value.
        iterations (int): Number of iterations.
        lr (float): Learning rate.

    Returns:
        numpy.ndarray: Low-dimensional representation of shape
            (n, ndims).
    """
    X_pca = pca(X, idims)

    n = X.shape[0]

    P = P_affinities(X_pca, perplexity=perplexity)

    Y = np.random.randn(n, ndims)

    dY = np.zeros((n, ndims))
    iY = np.zeros((n, ndims))

    for i in range(iterations):
        if i < 100:
            P_use = P * 4
        else:
            P_use = P

        grad, Q = grads(Y, P_use)

        if i < 20:
            alpha = 0.5
        else:
            alpha = 0.8

        iY = alpha * iY - lr * grad

        Y += iY

        Y -= np.mean(Y, axis=0)

        if (i + 1) % 100 == 0:
            C = cost(P, Q)
            print("Cost at iteration {}: {}".format(i + 1, C))

    return Y
