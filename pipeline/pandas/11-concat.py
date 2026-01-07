#!/usr/bin/env python3
""" Concat two df's """
index = __import__('10-index').index
import pandas as pd


def concat(df1, df2):
    """ Concat df1 and df2 """
    df1 = index(df1)
    df2 = index(df2)
    df2 = df2[df2.index <= 1417411920]
    return pd.concat([df1, df2], keys=['coinbase', 'bitstamp'])
