#!/usr/bin/env python3
""" Function for indexing df """


def index(df):
    """ Set Timestamp col as index of df """
    df.reindex(index=df['Timestamp'])
