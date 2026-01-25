#!/usr/bin/env python3
""" Find the transpose of 2D matrix """


def matrix_transpose(matrix):
    """ Return transpose of 2D matrix """
    for i in range(matrix):
        for j in range(i + 1, matrix[0]):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    return matrix
