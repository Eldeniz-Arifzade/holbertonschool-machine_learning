#!/usr/bin/env python3
""" This module will define a function which will concat arrays """


def cat_arrays(arr1, arr2):
    """ Function for concatinating arrays """
    new_array = arr1.copy()
    new_array.extend(arr2)
    return new_array
