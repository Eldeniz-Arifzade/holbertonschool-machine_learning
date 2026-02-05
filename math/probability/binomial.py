#!/usr/bin/env python3
""" This module will define a class Binomial """


class Binomial():
    """ Class for representing binomial distribution """
    def __init__(self, data=None, n=1, p=0.5):
        """ Initialize class """
        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')
            if p < 0 or p > 1:
                raise ValueError('p must be greater than 0 and less than 1')
            self.n = n
            self.p = p
        elif not isinstance(data, list):
            raise TypeError('data must be a list')
        elif len(data) < 2:
            raise ValueError('data must contain multiple values')
        else:
            mean = sum(data) / len(data)
            variance = 0
            for trail in data:
                variance += (trail - mean) ** 2
            variance = variance / len(data)
            p = 1 - variance / mean
            n = round(mean / p)
            p = mean / n
            self.p = p
            self.n = n
