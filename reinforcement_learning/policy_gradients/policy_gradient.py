#!/usr/bin/env python3
"""Compute action probabilities for a linear softmax policy."""
import numpy as np


def policy_gradient(state, weight):
    """Sample an action and compute its log-policy gradient.

    Args:
        state: One observation with shape (n,) or (1, n).
        weight: Weight matrix with shape (n, a).

    Returns:
        A tuple containing the sampled action and its gradient with
        respect to weight, with shape (n, a).
    """
    probabilities = policy(state.reshape(1, -1), weight)[0]
    action = np.random.choice(probabilities.size, p=probabilities)
    dlog = -probabilities
    dlog[action] += 1
    gradient = np.outer(state, dlog)
    return action, gradient
