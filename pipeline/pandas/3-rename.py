#!/usr/bin/env python3
""" Rename column """

import pandas as pd


def rename(df):
    """ Rename column """
    df.rename(columns={'Timestamp': 'Datetime'}, inplace=True)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    return df[['Datetime', 'Close']]