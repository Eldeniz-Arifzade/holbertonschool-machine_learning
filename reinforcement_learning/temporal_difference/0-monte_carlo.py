#!/usr/bin/env python3
"""Estimate state values using first-visit Monte Carlo prediction."""


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Evaluate a policy using sampled discounted returns.

    Args:
        env: A Gymnasium environment with integer state indices.
        V: A numpy.ndarray of shape (s,) containing state values.
        policy: A function mapping a state to an action.
        episodes: Number of episodes to sample.
        max_steps: Maximum number of steps per episode.
        alpha: Learning rate.
        gamma: Discount factor.

    Returns:
        The same array V, updated in place. Returns use rewards observed
        up to termination, truncation, or the max_steps limit.
    """
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []
        first_visit = {}

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))
            first_visit.setdefault(state, step)
            state = next_state

            if terminated or truncated:
                break

        G = 0.0
        for step in range(len(episode) - 1, -1, -1):
            state, reward = episode[step]
            G = reward + gamma * G
            if first_visit[state] == step:
                V[state] += alpha * (G - V[state])

    return V
