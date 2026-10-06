#!/usr/bin/env python3
"""Estimate state values using TD(lambda) with accumulating traces."""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """Evaluate a policy using temporal differences and eligibility traces.

    Args:
        env: A Gymnasium environment with integer state indices.
        V: A numpy.ndarray of shape (s,) containing state values.
        policy: A function mapping a state to an action.
        lambtha: Eligibility trace decay factor.
        episodes: Number of episodes to sample.
        max_steps: Maximum number of steps per episode.
        alpha: Learning rate.
        gamma: Discount factor.

    Returns:
        The same array V, updated in place. Terminal-state values are
        included in TD targets to match the project's sample behavior.
    """
    trace_decay = gamma * lambtha

    for _ in range(episodes):
        state, _ = env.reset()
        traces = np.zeros_like(V)

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            delta = reward + gamma * V[next_state] - V[state]
            traces[state] += 1
            V += alpha * delta * traces
            traces *= trace_decay
            state = next_state

            if terminated or truncated:
                break

    return V
