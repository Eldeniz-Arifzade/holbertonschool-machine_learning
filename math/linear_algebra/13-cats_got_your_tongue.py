#!/usr/bin/env python3
""" This module will define a function named np_cat """
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ Concat matrices along given axis """
    return np.concatenate((mat1, mat2), axis)
