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
            self.stddev = stddev ** (1 / 2)
