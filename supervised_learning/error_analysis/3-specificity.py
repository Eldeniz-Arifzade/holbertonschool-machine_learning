#!/usr/bin/env python3
""" Write a function for calculating specificity of confusion matrix """
import numpy as np


def specificity(confusion):
    """ Function for calculating specificity """
    classes = confusion.shape[0]
    arr = np.zeros(classes)
    for i in range(classes):
        tn_fp = classes - np.sum(confusion[i])
        tn = classes - confusion[i][i]
        arr[i] = tn_fp / tn
    return arr
