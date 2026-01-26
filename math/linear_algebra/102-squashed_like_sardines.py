#!/usr/bin/env python3
""" Advanced number #3 """


def cat_matrices(mat1, mat2, axis=0):
    """ Concant matrices with py """
    def shape(matrix):
        """ Shape of a matrix """
        dims = []
        while type(matrix) is list:
            dims.append(len(matrix))
            matrix = matrix[0]
        return tuple(dims)

    shape1 = shape(mat1)
    shape2 = shape(mat2)

    if axis < 0 or axis >= len(shape1) or axis >= len(shape2):
        return None

    for i in range(len(shape1)):
        if i != axis and shape1[i] != shape2[i]:
            return None

    if axis == 0:
        return mat1 + mat2

    result = []

    for i in range(len(mat1)):
        result.append(cat_matrices(mat1[i], mat2[i], axis - 1))

    return result
