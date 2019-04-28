import numpy as np
from numpy import power as Power

from numerical import derivative


def linear(x: np.array) -> np.array:
    """ One-dimensional linear spline.

    Spline defined on the interval (-1, 1).
    """
    x = np.array(x, dtype=np.float64)
    fx = np.zeros_like(x)
    idx = np.logical_and(x > -1, x <= 0)
    fx[idx] = x[idx] + 1
    idx = np.logical_and(x > 0, x < 1)
    fx[idx] = -x[idx] + 1
    idx = np.logical_or(x <= -1, x >= 1)
    fx[idx] = 0
    return fx


@derivative.setup(ftype="numerical")
def schoenberg(x: np.array) -> np.array:
    """ One-dimensional Schoenberg spline of 5-th order.

    Spline defined on the interval (0, 3)
    """
    fx = np.zeros_like(x)
    # if x >= 0 and x < 1
    idx = np.logical_and(x >= 0., x < 1.)
    fx[idx] = 0.55 - Power(x[idx], 2)/2. + Power(x[idx], 4)/4. - Power(x[idx], 5)/12.
    # if x >= 1 and x < 2
    idx = np.logical_and(x >= 1., x < 2.)
    fx[idx] = 0.425 + (5*x[idx])/8. - (7*Power(x[idx], 2))/4. + (5*Power(x[idx], 3))/4. -\
              (3*Power(x[idx], 4))/8. + Power(x[idx], 5)/24.
    # if x >=2 and x < 3
    idx = np.logical_and(x >= 2., x < 3.)
    fx[idx] = 2.025 - (27*x[idx])/8. + (9*Power(x[idx], 2))/4. - (3*Power(x[idx], 3))/4. +\
              Power(x[idx], 4)/8. - Power(x[idx], 5)/120.
    # if x < 0 and x > 3
    idx = np.logical_and(x < 0., x >= 3.)
    fx[idx] = 0
    return fx
