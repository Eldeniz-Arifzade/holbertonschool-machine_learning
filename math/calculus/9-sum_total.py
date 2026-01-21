#!/usr/bin/env python3
""" this module will define a function for summing squares """


def summation_i_squared(n):
    """ Sum of squares """
    if n < 1:
        return None
    if n == 1:
        return 1
    return n**2 + int(summation_i_squared(n-1))
