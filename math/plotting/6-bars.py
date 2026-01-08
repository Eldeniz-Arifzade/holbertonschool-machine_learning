#!/usr/bin/env python3
""" Stacking bars """

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """ Plot stacked bar graph """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4,3))
    plt.figure(figsize=(6.4, 4.8))

    # your code here
    x = ['Farrah', 'Fred', 'Felicia']
    y0 = fruit[0]
    y1 = fruit[1]
    y2 = fruit[2]
    y3 = fruit[3]

    plt.bar(x, y0, color='red', width=0.5, label='apples')
    plt.bar(x, y1, bottom=y0, color='yellow', width=0.5, label='bananas')
    plt.bar(x, y2, bottom=y0+y1, color='#ff8000', width=0.5, label='oranges')
    plt.bar(x, y3, bottom=y0+y1+y2, color='#ffe5b4', width=0.5, label='peaches')
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.legend()
    plt.yticks(range(0, 81, 10))
