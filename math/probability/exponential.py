#!/usr/bin/env python3
""" This module will define a class Exponential """


class Exponential():
    """ Class Exponential that represents an exponential distribution """
    def __init__(self, data=None, lambtha=1.):
        """ Initalize class """
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = lambtha
        elif not isinstance(data, list):
            raise TypeError('data must be a list')
        elif len(data) < 2:
            raise ValueError('data must contain multiple values')
        else:
            self.lambtha = sum(data) / len(data)
