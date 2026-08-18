#!/usr/bin/env python3
"""Q-learning training."""

import numpy as np


epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Train a Q-table using the Q-learning algorithm."""
    total_rewards = []

    for _ in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, terminated, truncated, _ = env.step(action)

            if terminated and reward == 0:
                reward = -1

            if terminated or truncated:
                target = reward
            else:
                target = reward + gamma * np.max(Q[new_state])

            Q[state, action] = (
                Q[state, action]
                + alpha * (target - Q[state, action])
            )

            state = new_state
            total_reward += reward

            if terminated or truncated:
                break

        total_rewards.append(total_reward)
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q, total_rewards
