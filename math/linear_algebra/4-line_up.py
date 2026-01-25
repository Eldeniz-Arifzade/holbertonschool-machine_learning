#!/usr/bin/env python3
""" Add two arrays element-wise """


def add_arrays(arr1, arr2):
    """ This function will add arrays elements-wise """
    if len(arr1) != len(arr2):
        return None
    arr = [(arr1[i] + arr2[i]) for i in range(len(arr1))]
    return arr
