#!/usr/bin/env python3
""" This module will define a mat_mul function """


def mat_mul(mat1, mat2):
    """ Perform matrix multiplication """
    if len(mat1[0]) != len(mat2):
        return None
    new_mat = [[0] * len(mat2[0])] * len(mat1)
    for i in range(len(mat1)):
        entry = 0
        for j in range(len(mat1[0])):
            entry += mat1[i][j] * mat2[j][i]
        new_mat[i][j] = entry
    return new_mat
