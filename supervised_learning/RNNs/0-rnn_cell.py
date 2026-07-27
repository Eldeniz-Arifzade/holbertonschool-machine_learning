#!/usr/bin/env python3
"""Defines a cell for a simple recurrent neural network."""

import numpy as np


class RNNCell:
    """Represents one cell of a simple recurrent neural network."""

    def __init__(self, i, h, o):
        """Initialize the cell's weights and biases.

        Args:
            i: Dimensionality of the input data.
            h: Dimensionality of the hidden state.
            o: Dimensionality of the output.
        """
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step.

        Args:
            h_prev: Previous hidden state of shape ``(m, h)``.
            x_t: Input data for the time step of shape ``(m, i)``.

        Returns:
            The next hidden state and the softmax output.
        """
        cell_input = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(cell_input, self.Wh) + self.bh)

        logits = np.matmul(h_next, self.Wy) + self.by
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        y = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return h_next, y
