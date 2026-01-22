#!/usr/bin/env python3
""" This module will define a function for integrating """


def poly_integral(poly, C=0):
    """ Function for integrating polynomial """
    if not isinstance(poly, list) or not isinstance(C, int) or len(poly) == 0:
        return None
    poly_integral = [C]
    for idx, power in enumerate(poly):
        element = power / (idx + 1)
        if element == int(element):
            poly_integral.append(int(element))
        else:
            poly_integral.append(element)
    if poly_integral[-1] == 0:
        poly_integral.pop()
    return poly_integral
