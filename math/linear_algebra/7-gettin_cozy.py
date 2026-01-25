#!/usr/bin/env python3
""" Concat two matrices """


def cat_matrices2D(mat1, mat2, axis=0):
    """ Concantinate two 2D matrices """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2
    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        mat = mat1.copy()
        for i in range(len(mat1)):
            mat[i].extend(mat2[i])
        return mat
