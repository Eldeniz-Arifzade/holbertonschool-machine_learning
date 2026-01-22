#!/usr/bin/env python3
""" This module will define a function for integrating """


def poly_integral(poly, C=0):
    """ Function for integrating polynomial """
    if not (isinstance(poly, list) or len(poly) == 0 or isinstance(C, int)):
        return None
    poly_integral = [C]
    for idx, power in enumerate(poly):
        poly_integral.append(poly[idx] / (power + 1))
    return poly_integral
