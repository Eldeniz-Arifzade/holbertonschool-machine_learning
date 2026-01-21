#!/usr/bin/env python3
""" this module will define a function for summing squares """


def summation_i_squared(n):
    """ Sum of squares """
    if n < 1:
        return None
    return n * (n + 1) * (2 * n + 1) / 6
