#!/usr/bin/env python3
""" This module will define a function which will concat arrays """


def cat_arrays(arr1, arr2):
    """ Function for concatinating arrays """
    arr1.extend(arr2)
    return arr1
