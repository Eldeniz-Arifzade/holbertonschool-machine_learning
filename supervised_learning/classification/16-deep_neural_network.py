#!/usr/bin/env python3
""" This module will define a class named DeepNeuralNetwork """
import numpy as np


class DeepNeuralNetwork():
    """ Class for implementing L layered NN """
    def __init__(self, nx, layers):
        """ Initialize class """
        # validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # validate layers
        if (not isinstance(layers, list)) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        for nodes in layers:
            if not isinstance(nodes, int) or nodes <= 0:
                raise TypeError("layers must be a list of positive integers")

        # attributes
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        # initialize weights (He initialization)
        prev = nx

        for l in range(self.L):
            nodes = layers[l]

            self.weights["W{}".format(l+1)] = (
                np.random.randn(nodes, prev) * np.sqrt(2 / prev)
            )

            self.weights["b{}".format(l+1)] = np.zeros((nodes, 1))

            prev = nodes
