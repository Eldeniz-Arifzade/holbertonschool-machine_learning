#!/usr/bin/env python3
"""Compute a softmax policy and its log-probability gradient."""
import numpy as np


def policy(matrix, weight):
    """Compute action probabilities from states and weights."""
    scores = np.matmul(matrix, weight)
    scores -= np.max(scores, axis=-1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    return probabilities


def policy_gradient(state, weight):
    """Sample an action and return its log-policy gradient.

    Args:
        state: Observation with shape (n,) or (1, n).
        weight: Weight matrix with shape (n, a).

    Returns:
        The sampled action and a gradient with shape (n, a).
    """
    probabilities = policy(state.reshape(1, -1), weight)[0]
    action = np.random.choice(probabilities.size, p=probabilities)
    dlog = -probabilities
    dlog[action] += 1
    gradient = np.outer(state, dlog)
    return action, gradient
