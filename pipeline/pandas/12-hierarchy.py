#!/usr/bin/env python3
""" This module will define a function for concatinating 2dfs """
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """ Concat df1 and df2 """
    df1 = index(df1)
    df2 = index(df2)
    df1 = df1[(df1.index >= 1417411980) & (df1.index <= 1417417980)]
    df2 = df2[(df2.index >= 1417411980) & (df2.index <= 1417417980)]
    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
    df = df.swaplevel(0, 1)
    df = df.sort_index(level=0)
    return df
