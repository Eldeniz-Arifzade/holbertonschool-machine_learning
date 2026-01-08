#!/usr/bin/env python3
""" This module will define a line function """

import numpy as np
import matplotlib.pyplot as plt


def line():
    """ Plot y and limit x between 0 and 10 """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    # your code here
    plt.plot(y)
    plt.xlim(0, 10)
