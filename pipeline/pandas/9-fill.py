#!/usr/bin/env python3
""" This module will define a fill function """


def fill(df):
    """ Function for properly filling missing values """
    df = df.drop(['Weighted_Price'], axis=1)
    df['Close'] = df['Close'].fillna(method='ffill', axis=0)
    for col in ['High', 'Low', 'Open']:
        df[col] = df[col].fillna(df['Close'])
    cols = ['Volume_(BTC)', 'Volume_(Currency)']
    df[cols] = df[cols].fillna(0)
    return df
