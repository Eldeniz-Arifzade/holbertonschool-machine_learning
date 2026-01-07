#!/usr/bin/env python3
""" Function for indexing df """


def index(df):
    """ Set Timestamp col as index of df """
    return df.set_index('Timestamp')
