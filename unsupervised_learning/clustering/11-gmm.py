#!/usr/bin/env python3
"""Calculates a GMM from a dataset using sklearn."""
import sklearn.mixture


def gmm(X, k):
    """Calculate a GMM from a dataset.

    Args:
        X (numpy.ndarray): shape (n, d) containing the dataset
        k (int): number of clusters

    Returns:
        pi: cluster priors of shape (k,)
        m: centroid means of shape (k, d)
        S: covariance matrices of shape (k, d, d)
        clss: cluster indices for each data point of shape (n,)
        bic: BIC value for the model
    """
    model = sklearn.mixture.GaussianMixture(n_components=k)
    model.fit(X)
    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)
    return pi, m, S, clss, bic
