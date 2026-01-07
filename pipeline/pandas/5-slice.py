#!/usr/bin/env python3
""" This module will define a function for slicing df """

import pandas as pd


def slice(df):
    """ Select every 60th row"""
    return df[['High', 'Low', 'Close', 'Volume_BTC']].iloc[::60, :]
