#!/usr/bin/env python3
""" Add two 2D matrices elements wise """


def add_matrices2D(mat1, mat2):
    """ This function will add two 2D matrices """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    return [
        [
            (mat1[i][j] + mat2[i][j]) for j in range(len(mat1[0]))
        ] for i in range(mat1)
    ]
