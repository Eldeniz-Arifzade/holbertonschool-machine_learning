#!/usr/bin/env python3
""" Function for plotting hist graph """

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """ Plot hist graph """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # your code here
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.title('Project A')
