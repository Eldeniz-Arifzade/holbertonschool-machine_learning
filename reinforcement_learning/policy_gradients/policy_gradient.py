#!/usr/bin/env python3
"""Compute action probabilities for a linear softmax policy."""
import numpy as np


def policy(matrix, weight):
    """Compute a numerically stable softmax policy.

    Args:
        matrix: State matrix with shape (m, n).
        weight: Weight matrix with shape (n, a).

    Returns:
        Action probabilities with shape (m, a).
    """
    scores = np.matmul(matrix, weight)
    scores -= np.max(scores, axis=-1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    return probabilities
