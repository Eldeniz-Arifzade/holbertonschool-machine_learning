#!/usr/bin/env python3
"""Train a Q table using SARSA(lambda) and accumulating traces."""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """Update Q using epsilon-greedy SARSA with eligibility traces.

    Args:
        env: A Gymnasium environment with integer state indices.
        Q: A numpy.ndarray of shape (s, a) containing action values.
        lambtha: Eligibility trace decay factor.
        episodes: Number of training episodes.
        max_steps: Maximum number of steps per episode.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Initial exploration probability.
        min_epsilon: Minimum exploration probability.
        epsilon_decay: Exponential exploration decay rate.

    Returns:
        The same Q table, updated in place. Terminal action values are
        included in TD targets to match the project's sample behavior.
    """
    initial_epsilon = epsilon
    trace_decay = gamma * lambtha
    n_actions = Q.shape[1]

    def epsilon_greedy(state):
        """Choose an action using the current exploration probability."""
        if np.random.uniform() < epsilon:
            return np.random.randint(n_actions)
        return np.argmax(Q[state])

    for episode in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy(state)
        traces = np.zeros_like(Q)

        for _ in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = epsilon_greedy(next_state)

            delta = reward + gamma * Q[next_state, next_action]
            delta -= Q[state, action]
            traces[state, action] += 1
            Q += alpha * delta * traces
            traces *= trace_decay

            state, action = next_state, next_action
            if terminated or truncated:
                break

        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * np.exp(
            -epsilon_decay * episode)

    return Q
