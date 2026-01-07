#!/usr/bin/env python3
""" This module will define a function which will sort by specified col"""


def high(df):
    """ Sort by High price in descending order """
    return df.sort_values('High', ascending=False)
