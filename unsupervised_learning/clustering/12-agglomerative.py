#!/usr/bin/env python3
"""Performs agglomerative clustering"""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Performs agglomerative clustering on a dataset

    Args:
        X (numpy.ndarray): shape (n, d) containing the dataset
        dist (float): maximum cophenetic distance

    Returns:
        numpy.ndarray: shape (n,) containing cluster indices
    """
    linkage = scipy.cluster.hierarchy.linkage(X, method='ward')

    scipy.cluster.hierarchy.dendrogram(
        linkage,
        color_threshold=dist
    )

    plt.show()

    return scipy.cluster.hierarchy.fcluster(
        linkage,
        t=dist,
        criterion='distance'
    )
