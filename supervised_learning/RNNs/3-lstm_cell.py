#!/usr/bin/env python3
"""Defines a long short-term memory cell."""

import numpy as np


class LSTMCell:
    """Represents one long short-term memory unit."""

    def __init__(self, i, h, o):
        """Initialize the cell's weights and biases.

        Args:
            i: Dimensionality of the input data.
            h: Dimensionality of the hidden and cell states.
            o: Dimensionality of the output.
        """
        self.Wf = np.random.randn(h + i, h)
        self.Wu = np.random.randn(h + i, h)
        self.Wc = np.random.randn(h + i, h)
        self.Wo = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def _sigmoid(value):
        """Return the element-wise sigmoid of ``value``."""
        return 1 / (1 + np.exp(-value))

    def forward(self, h_prev, c_prev, x_t):
        """Perform forward propagation for one time step.

        Args:
            h_prev: Previous hidden state of shape ``(m, h)``.
            c_prev: Previous cell state of shape ``(m, h)``.
            x_t: Input data for the time step of shape ``(m, i)``.

        Returns:
            The next hidden state, next cell state, and softmax output.
        """
        cell_input = np.concatenate((h_prev, x_t), axis=1)

        forget = self._sigmoid(
            np.matmul(cell_input, self.Wf) + self.bf
        )
        update = self._sigmoid(
            np.matmul(cell_input, self.Wu) + self.bu
        )
        candidate = np.tanh(
            np.matmul(cell_input, self.Wc) + self.bc
        )
        output = self._sigmoid(
            np.matmul(cell_input, self.Wo) + self.bo
        )

        c_next = forget * c_prev + update * candidate
        h_next = output * np.tanh(c_next)

        logits = np.matmul(h_next, self.Wy) + self.by
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        y = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return h_next, c_next, y
