#!/usr/bin/env python3
"""Module for Bayesian Optimization implementation."""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


def optimize(self, iterations=100):
    """
    Optimizes the black-box function

    Args:
        iterations: maximum number of iterations

    Returns:
        X_opt, Y_opt
    """
    for _ in range(iterations):
        X_next, _ = self.acquisition()

        if np.any(X_next == self.gp.X):
            break

        Y_next = self.f(X_next)

        self.gp.update(X_next.reshape(1, 1),
                       Y_next.reshape(1, 1))

    if self.minimize:
        idx = np.argmin(self.gp.Y)
    else:
        idx = np.argmax(self.gp.Y)

    X_opt = self.gp.X[idx]
    Y_opt = self.gp.Y[idx]

    return X_opt, Y_opt
