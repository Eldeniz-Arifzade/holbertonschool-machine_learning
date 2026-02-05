#!/usr/bin/env python3
""" This module will define a class for normal distribution """


class Normal():
    """ Class for representing normal distribution """
    def __init__(self, data=None, mean=0., stddev=1.):
        """ Initialize class """
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            self.stddev = stddev
            self.mean = mean
        elif not isinstance(data, list):
            raise TypeError('data must be a list')
        elif len(data) < 2:
            raise ValueError('data must contain multiple values')
        else:
            self.mean = sum(data) / len(data)
            stddev = 0
            for i in data:
                stddev += (self.mean - i) ** 2
            self.stddev = (stddev / len(data)) ** (1 / 2)

    def z_score(self, x):
        """ Calculate z score """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """ Calculate x from z score """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """ Calculate PDF value for given x """
        return (1 / ((2 * 3.1415926536) ** (1 / 2) * self.stddev)
               * 2.7182818285 ** ((-(x - self.mean) ** 2) / (2 * self.stddev ** 2))
               )
