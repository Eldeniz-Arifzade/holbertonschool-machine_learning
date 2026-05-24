#!/usr/bin/env python3
"""Performs K-means on a dataset"""

import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means

    Args:
        X (numpy.ndarray): dataset of shape (n, d)
        k (int): number of clusters

    Returns:
        numpy.ndarray: initialized centroids of shape (k, d)
        None: on failure
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(k, int) or k <= 0):
        return None

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    return np.random.uniform(min_vals, max_vals, (k, X.shape[1]))


def kmeans(X, k, iterations=1000):
    """Performs K-means clustering

    Args:
        X (numpy.ndarray): dataset of shape (n, d)
        k (int): number of clusters
        iterations (int): maximum number of iterations

    Returns:
        tuple:
            C (numpy.ndarray): centroid means of shape (k, d)
            clss (numpy.ndarray): cluster indices for each point
        (None, None): on failure
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(k, int) or k <= 0 or
            not isinstance(iterations, int) or iterations <= 0):
        return None, None

    n, d = X.shape

    C = initialize(X, k)
    if C is None:
        return None, None

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    for i in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        new_C = np.copy(C)

        for j in range(k):
            points = X[clss == j]

            if len(points) == 0:
                new_C[j] = np.random.uniform(min_vals, max_vals)
            else:
                new_C[j] = np.mean(points, axis=0)

        if np.array_equal(C, new_C):
            return C, clss

        C = new_C

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(distances, axis=1)

    return C, clss
