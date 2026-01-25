#!/usr/bin/env python3
""" Advanced task #2 """


def add_matrices(mat1, mat2):
    """ Add 2 matrices """
    result = []
    if type(mat1) is list and type(mat2) is list:
        if len(mat1) == len(mat2):
            for i in range(len(mat1)):
                sub = add_matrices(mat1[i], mat2[i])
                if sub is None:
                    return None
                result.append(sub)
            return result
        else:
            return None
    else:
        return mat1 + mat2
