#!/usr/bin/env python3
""" This module will define a function which will return shape of matrix """


shape = []
def matrix_shape(matrix):
    """ Calculate the shape of the matrix """
    shape.append(len(matrix))
    if type(matrix[0]) == list:
        return matrix_shape(matrix[0])
    return shape
