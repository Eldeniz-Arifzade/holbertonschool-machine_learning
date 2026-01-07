#!/usr/bin/env python3
""" Flip it and Switch it """


def flip_switch(df):
    """ Reverse and return transpose of df """
    df.sort_index(ascending=False, inplace=True)
    return df.T
