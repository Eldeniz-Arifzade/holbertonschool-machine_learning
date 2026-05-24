#!/usr/bin/env python3
"""Calculates a GMM from a dataset using sklearn"""

import sklearn.mixture


def gmm(X, k):
    """Calculates a GMM from a dataset

    Args:
        X (numpy.ndarray): shape (n, d) containing the dataset
        k (int): number of clusters

    Returns:
        tuple:
            pi (numpy.ndarray): cluster priors
            m (numpy.ndarray): centroid means
            S (numpy.ndarray): covariance matrices
            clss (numpy.ndarray): cluster labels
            bic (float): BIC value for the model
    """
    gmm = sklearn.mixture.GaussianMixture(n_components=k)

    gmm.fit(X)

    return (gmm.weights_,
            gmm.means_,
            gmm.covariances_,
            gmm.predict(X),
            gmm.bic(X))
