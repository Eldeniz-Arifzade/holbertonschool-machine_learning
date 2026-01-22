#!/usr/bin/env python3
""" This module will define a function for integrating """


def poly_integral(poly, C=0):
    """ Function for integrating polynomial """
    if not (isinstance(poly, list) or len(poly) == 0):
        return None
    poly = [0]
    for idx, power in enumerate(poly):
        poly.append(poly[idx] / (power + 1))
    return poly    