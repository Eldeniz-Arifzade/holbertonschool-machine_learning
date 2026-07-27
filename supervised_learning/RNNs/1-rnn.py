#!/usr/bin/env python3
"""Performs forward propagation through a simple RNN."""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """Perform forward propagation for a simple recurrent neural network.

    Args:
        rnn_cell: Cell used for forward propagation.
        X: Input data of shape ``(t, m, i)``.
        h_0: Initial hidden state of shape ``(m, h)``.

    Returns:
        A tuple ``(H, Y)`` containing every hidden state, including
        ``h_0``, and the output at every time step.
    """
    time_steps, batch_size, _ = X.shape
    hidden_size = h_0.shape[1]
    output_size = rnn_cell.by.shape[1]

    H = np.empty((time_steps + 1, batch_size, hidden_size))
    Y = np.empty((time_steps, batch_size, output_size))
    H[0] = h_0

    for time_step in range(time_steps):
        H[time_step + 1], Y[time_step] = rnn_cell.forward(
            H[time_step], X[time_step]
        )

    return H, Y
