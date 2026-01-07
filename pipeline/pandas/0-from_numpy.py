#!/usr/bin/env python3

''' This module will define a function for creating a df from array '''


import pandas as pd
def from_numpy(array):
    """ Create dataframe from numpy array """
    cols = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    return pd.DataFrame(array, columns=list(cols[array.shape[1]]))