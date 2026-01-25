#!/usr/bin/env python3
""" This module will define a func named np_elementwise """


def np_elementwise(mat1, mat2):
    """ Conduct elementwise operations """
    return (
        mat1 + mat2,
        mat1 - mat2,
        mat1 * mat2,
        mat1 / mat2
    )
