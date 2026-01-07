#!/usr/bin/env python3
""" Load df from a file """

import pandas as pd


def from_file(filename, delimiter):
    """ Load df from a file """
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
