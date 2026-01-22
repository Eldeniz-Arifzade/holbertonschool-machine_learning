#!/usr/bin/env python3
""" This module will define a function for integrating """


def poly_integral(poly, C=0):
    """ Function for integrating polynomial """
    if not (isinstance(poly, list) or isinstance(C, int)):
        return None
    poly_integral = [C]
    for idx, power in enumerate(poly):
        element = power / (idx + 1)
        if element == int(element):
            poly_integral.append(int(element))
        else:
            poly_integral.append(element)
    return poly_integral
