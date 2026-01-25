#!/usr/bin/env python3
""" Concat two matrices """


def cat_matrices2D(mat1, mat2, axis=0):
    """ Concantinate two 2D matrices """
    if axis == 0:
        return mat1 + mat2
    if axis == 1:
        mat = mat1.copy()
        for i in range(len(mat1)):
            mat[i].extend(mat2[i])
        return mat
