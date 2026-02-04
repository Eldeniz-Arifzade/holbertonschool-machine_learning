#!/usr/bin/env python3
""" This module will define a class named Poisson """


class Poisson():
    """ This class will demonstrate a poisson distribution """
    def __init__(self, data=None, lambtha=1.):
        """ Function for initializing class """
        if data is None:
            self.data = lambtha
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
        elif isinstance(data, list):
            raise TypeError('data must be a list')
        elif len(data) < 2:
            raise ValueError('data must contain multiple values')
        else:
            lambtha = sum(data) / len(data)
