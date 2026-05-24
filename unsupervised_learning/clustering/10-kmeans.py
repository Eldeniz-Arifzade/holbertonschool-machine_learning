#!/usr/bin/env python3
"""Performs K-means clustering using sklearn"""

import sklearn.cluster


def kmeans(X, k):
    """Performs K-means on a dataset

    Args:
        X (numpy.ndarray): shape (n, d) containing the dataset
        k (int): number of clusters

    Returns:
        tuple:
            C (numpy.ndarray): centroid means
            clss (numpy.ndarray): cluster labels
    """
    kmeans = sklearn.cluster.KMeans(n_clusters=k)

    kmeans.fit(X)

    return kmeans.cluster_centers_, kmeans.labels_
