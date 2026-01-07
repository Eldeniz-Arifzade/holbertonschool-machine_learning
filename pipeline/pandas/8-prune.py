#!/usr/bin/env python3
""" This module will define a function which will remove NaN's from Close """


def prune(df):
    """ Remoce NaN entries from Close col """
    return df.dropna(subset=['Close'])
