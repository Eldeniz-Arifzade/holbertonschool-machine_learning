#!/usr/bin/env python3
""" Analyze dataframe """


def analyze(df):
    """ Function for describing stats of df """
    df.drop('Timestamp', axis=1, inplace=True)
    return df.describe()
