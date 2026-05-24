#!/usr/bin/env python3
"""Performs expectation maximization for a GMM"""

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000,
                             tol=1e-5, verbose=False):
    """Performs the EM algorithm for a GMM

    Args:
        X (numpy.ndarray): shape (n, d) containing dataset
        k (int): number of clusters
        iterations (int): max number of iterations
        tol (float): tolerance for early stopping
        verbose (bool): whether to print log likelihoods

    Returns:
        tuple:
            pi (numpy.ndarray): shape (k,) priors
            m (numpy.ndarray): shape (k, d) means
            S (numpy.ndarray): shape (k, d, d) covariance matrices
            g (numpy.ndarray): shape (k, n) posterior probabilities
            l (float): log likelihood
        (None, None, None, None, None): on failure
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(k, int) or k <= 0 or
            not isinstance(iterations, int) or iterations <= 0 or
            not isinstance(tol, float) or tol < 0 or
            not isinstance(verbose, bool)):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)

    if pi is None:
        return None, None, None, None, None

    g, l = expectation(X, pi, m, S)

    if g is None:
        return None, None, None, None, None

    for i in range(iterations):
        if verbose and (i % 10 == 0):
            print("Log Likelihood after {} iterations: {:.5f}"
                  .format(i, l))

        pi, m, S = maximization(X, g)

        if pi is None:
            return None, None, None, None, None

        g, new_l = expectation(X, pi, m, S)

        if g is None:
            return None, None, None, None, None

        if abs(new_l - l) <= tol:
            l = new_l

            if verbose:
                print("Log Likelihood after {} iterations: {:.5f}"
                      .format(i + 1, l))
            break

        l = new_l

    else:
        if verbose:
            print("Log Likelihood after {} iterations: {:.5f}"
                  .format(iterations, l))

    return pi, m, S, g, l
