#!/usr/bin/env python3
""" Advanced task #1 """


def np_slice(matrix, axes={}):
    """ Slice matrix """
    slices = [slice(None)] * matrix.ndim
    for key, value in axes.items():
        slices[key] = slice(*value)
    return matrix[tuple(slices)]
