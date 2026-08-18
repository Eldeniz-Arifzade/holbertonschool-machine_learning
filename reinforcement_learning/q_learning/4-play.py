#!/usr/bin/env python3
"""Play a FrozenLake episode using a trained Q-table."""

import numpy as np


def play(env, Q, max_steps=100):
    """Play one episode using the learned Q-table."""
    state, _ = env.reset()
    rendered_outputs = [env.render()]
    total_reward = 0

    for _ in range(max_steps):
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        rendered_outputs.append(env.render())

        if terminated or truncated:
            break

    return total_reward, rendered_outputs
