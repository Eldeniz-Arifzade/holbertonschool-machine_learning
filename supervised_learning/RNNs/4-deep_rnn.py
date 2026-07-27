#!/usr/bin/env python3
"""Performs forward propagation through a deep RNN."""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Perform forward propagation for a deep recurrent neural network.

    Args:
        rnn_cells: List of recurrent cells, one for each layer.
        X: Input data of shape ``(t, m, i)``.
        h_0: Initial hidden states of shape ``(l, m, h)``.

    Returns:
        A tuple ``(H, Y)`` containing all hidden states, including the
        initial states, and the final layer's outputs at each time step.
    """
    time_steps, batch_size, _ = X.shape
    layers, _, hidden_size = h_0.shape
    output_size = rnn_cells[-1].by.shape[1]

    H = np.empty(
        (time_steps + 1, layers, batch_size, hidden_size)
    )
    Y = np.empty((time_steps, batch_size, output_size))
    H[0] = h_0

    for time_step in range(time_steps):
        layer_input = X[time_step]

        for layer in range(layers):
            h_next, y = rnn_cells[layer].forward(
                H[time_step, layer], layer_input
            )
            H[time_step + 1, layer] = h_next
            layer_input = h_next

        Y[time_step] = y

    return H, Y
