#!/usr/bin/env python3
""" This module will define a function named np_cat """


def np_cat(mat1, mat2, axis=0):
    """ Concat matrices along given axis """
    return np.stack([mat1, mat2], axis)
