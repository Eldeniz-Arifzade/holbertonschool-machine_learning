#!/usr/bin/env python3
""" From DF to np array """

def array(df):
    """ Return last 10 rows in np array format """
    new_df = df[['High', 'Close']].tail(10)
    return new_df.to_numpy()
