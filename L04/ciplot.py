#!/usr/bin/python

"""
Sudhanva Sreesha
ssreesha@umich.edu
26-Jan-2018

Plot a confidence interval plot.
"""

import numpy as np
from matplotlib import pyplot as plt


def ciplot(t, mu, minus_sigma, plus_sigma, x_real, color=None):
    """
    Plots a shaded region on a graph between specified lower and upper confidence intervals (L and U).

    :param t: The time series corresponding to the state.
    :param mu: The predicted state of the variable.
    :param minus_sigma: THe lower bound of the confidence interval.
    :param plus_sigma: The upper bound of the confidence interval.
    :param x_real: The real value of the state variable.
    :param color: Color of the fill inside the lower and upper bound curves (optional).
    :return handle: The handle to the plot of the state variable.
    """

    assert minus_sigma.shape[0] == plus_sigma.shape[0]
    assert t.shape[0] == mu.shape[0]

    plt.fill_between(t, minus_sigma, plus_sigma, color=color, alpha=0.5)
    x_pred, = plt.plot(t, mu)
    x_real, = plt.plot(t, x_real)

    return x_pred, x_real
