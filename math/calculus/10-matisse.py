#!/usr/bin/env python3
""" Derivative of polynomial """


def poly_derivative(poly):
    """ Function for calculating derivative of polynomial """
    d_poly = []
    for idx, power in enumerate(poly):
        d_poly.append(idx * power)
    return d_poly[1:]
