#!/usr/bin/env python3


def high(df):
    """ Sort by High price in descending order """
    return df.sort_values('High', ascending=False)
