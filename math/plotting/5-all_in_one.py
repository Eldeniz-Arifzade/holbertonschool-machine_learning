#!/usr/bin/env python3
""" All graphs in one function """
import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """ Plot several graphs """
    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    # your code here
    fig = plt.figure()
    fig.suptitle('All in One')

    ax0 = plt.subplot2grid((3, 2), (0, 0))
    ax1 = plt.subplot2grid((3, 2), (0, 1))
    ax2 = plt.subplot2grid((3, 2), (1, 0))
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)

    ax0.plot(y0, color='red')
    ax0.xlim(0, 10)

    ax1.scatter(x1, y1, color='magenta')
    ax1.title("Men's Height vs Weight", fontsize='x-small')
    ax1.xlabel('Height (in)', fontsize='x-small')
    ax1.ylabel('Weight (lbs)', fontsize='x-small')

    ax2.plot(x2, y2)
    ax2.xlabel('Time (years)', fontsize='x-small')
    ax2.ylabel('Fraction Remaining', fontsize='x-small')
    ax2.title('Exponential Decay of C-14', fontsize='x-small')
    ax2.yscale('log')
    ax2.xlim(0, 28650)  

    ax3.plot(x3, y31, linestyle='dashed', color='red', label='C-14')
    ax3.plot(x3, y32, linestyle='solid', color='green', label='Ra-226')
    ax3.xlabel('Time (years)', fontsize='x-small')
    ax3.ylabel('Fraction Remaining', fontsize='x-small')
    ax3.title('Exponential Decay of Radioactive Elements', fontsize='x-small')
    ax3.xlim(0, 20000)
    ax3.ylim(0, 1)
    ax3.legend()

    ax4.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
    ax4.xlabel('Grades', fontsize='x-small')
    ax4.ylabel('Number of Students', fontsize='x-small')
    ax4.xlim(0, 100)
    ax4.ylim(0, 30)
    ax4.xticks(range(0, 101, 10))
    ax4.title('Project A', fontsize='x-small')

    plt.tight_layout()
    plt.show()
