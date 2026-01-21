#!/usr/bin/env python3
""" Derivative of polynomial """


def poly_derivative(poly):
    """ Function for calculating derivative of polynomial """
    d_poly = []
    if len(poly) == 0:
        return None
    for idx, power in enumerate(poly):
        d_poly.append(idx * power)
    if len(d_poly) == 1:
        return [0]
    return d_poly[1:]
