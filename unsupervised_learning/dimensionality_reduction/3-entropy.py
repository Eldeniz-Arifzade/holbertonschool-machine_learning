#!/usr/bin/env python3
"""Entropy calculation for t-SNE."""

import numpy as np


def HP(Di, beta):
    """Calculates Shannon entropy and P affinities for a data point.

    Args:
        Di (numpy.ndarray): Array of shape (n - 1,) containing
            pairwise distances from one point to all other points.
        beta (numpy.ndarray): Array of shape (1,) containing the
            beta value for the Gaussian distribution.

    Returns:
        tuple:
            Hi (float): Shannon entropy of the points.
            Pi (numpy.ndarray): P affinities of shape (n - 1,).
    """
    Pi = np.exp(-Di * beta)

    sum_Pi = np.sum(Pi)

    Pi = Pi / sum_Pi

    Hi = -np.sum(Pi * np.log2(Pi))

    return Hi, Pi
