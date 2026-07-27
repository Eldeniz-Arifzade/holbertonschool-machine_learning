#!/usr/bin/env python3
"""Defines a gated recurrent unit cell."""

import numpy as np


class GRUCell:
    """Represents one gated recurrent unit."""

    def __init__(self, i, h, o):
        """Initialize the cell's weights and biases.

        Args:
            i: Dimensionality of the input data.
            h: Dimensionality of the hidden state.
            o: Dimensionality of the output.
        """
        self.Wz = np.random.randn(h + i, h)
        self.Wr = np.random.randn(h + i, h)
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def _sigmoid(value):
        """Return the element-wise sigmoid of ``value``."""
        return 1 / (1 + np.exp(-value))

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step.

        Args:
            h_prev: Previous hidden state of shape ``(m, h)``.
            x_t: Input data for the time step of shape ``(m, i)``.

        Returns:
            The next hidden state and the softmax output.
        """
        cell_input = np.concatenate((h_prev, x_t), axis=1)
        update = self._sigmoid(
            np.matmul(cell_input, self.Wz) + self.bz
        )
        reset = self._sigmoid(
            np.matmul(cell_input, self.Wr) + self.br
        )

        candidate_input = np.concatenate((reset * h_prev, x_t), axis=1)
        candidate = np.tanh(
            np.matmul(candidate_input, self.Wh) + self.bh
        )
        h_next = (1 - update) * h_prev + update * candidate

        logits = np.matmul(h_next, self.Wy) + self.by
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        y = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return h_next, y
