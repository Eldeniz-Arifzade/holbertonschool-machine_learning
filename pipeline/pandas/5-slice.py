#!/usr/bin/env python3
""" This module will define a function for slicing df """


def slice(df):
    """ Select every 60th row"""
    return df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60, :]
