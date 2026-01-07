#!/usr/bin/env python3
""" This module will define a fill function """


def fill(df):
    """ Function for properly filling missing values """
    df = df.drop(['Weighted_Price'], axis=1)
    df['Close'] = df['Close'].fillna(method='ffill', axis=0)
    df[['High', 'Low', 'Open']] = df[['High', 'Low', 'Open']].fillna(df['Close'])
    df[['Volume_(BTC)', 'Volume_(Currency)']].fillna(value=0, inplace=True)
    return df
