#!/usr/bin/env python3
""" This module will define a mat_mul function """


def mat_mul(mat1, mat2):
    """ Perform matrix multiplication """
    if len(mat1[0]) != len(mat2):
        return None
    new_mat = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]
    for i in range(len(mat1)):
        for k in range(len(mat2[0])):
            for j in range(len(mat1[0])):
                new_mat[i][k] += mat1[i][j] * mat2[j][k]
    return new_mat
