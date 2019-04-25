import numpy as np
from numpy import power as Power


class shengerg_spline_derivatives:
    def __init__(self, max_order=3):
        if max_order > 3:
            raise NotImplementedError(f"Derivative os order {max_order} for Shenberg splines is not implemented. "
                                      f"Max order of available derivative is 3.")
        self._max_order = max_order

    def __call__(self, f):
        for i in range(self._max_order):
            f.__dict__[f'd{i + 1}'] = shengerg_spline_derivatives.__dict__[f'd{i + 1}'].__func__
        return f

    @staticmethod
    def d1(x: np.array) -> np.array:
        fx = np.zeros_like(x)
        # if x >= 0 and x < 1
        idx = np.logical_and(x >= 0., x < 1.)
        fx[idx] = -x[idx] + Power(x[idx], 3) - (5 * Power(x[idx], 4)) / 12.
        # if x >= 1 and x < 2
        idx = np.logical_and(x >= 1., x < 2.)
        fx[idx] = 0.625 - (7 * x[idx]) / 2. + (15 * Power(x[idx], 2)) / 4. -\
                  (3 * Power(x[idx], 3)) / 2. + (5 * Power(x[idx], 4)) / 24.
        # if x >=2 and x < 3
        idx = np.logical_and(x >= 2., x < 3.)
        fx[idx] = -3.375 + (9 * x[idx]) / 2. - (9 * Power(x[idx], 2)) / 4. +\
                  Power(x[idx], 3) / 2. - Power(x[idx], 4) / 24.
        # if x < 0 and x > 3
        idx = np.logical_and(x < 0., x >= 3.)
        fx[idx] = 0
        return fx

    @staticmethod
    def d2(x: np.array) -> np.array:
        fx = np.zeros_like(x)
        # if x >= 0 and x < 1
        idx = np.logical_and(x >= 0., x < 1.)
        fx[idx] = -1 + 3 * Power(x[idx], 2) - (5 * Power(x[idx], 3)) / 3.
        # if x >= 1 and x < 2
        idx = np.logical_and(x >= 1., x < 2.)
        fx[idx] = -3.5 + (15 * x[idx]) / 2. - (9 * Power(x[idx], 2)) / 2. + (5 * Power(x[idx], 3)) / 6.
        # if x >=2 and x < 3
        idx = np.logical_and(x >= 2., x < 3.)
        fx[idx] = 4.5 - (9 * x[idx]) / 2. + (3 * Power(x[idx], 2)) / 2. - Power(x[idx], 3) / 6.
        # if x < 0 and x > 3
        idx = np.logical_and(x < 0., x >= 3.)
        fx[idx] = 0
        return fx

    @staticmethod
    def d3(x: np.array) -> np.array:
        fx = np.zeros_like(x)
        # if x >= 0 and x < 1
        idx = np.logical_and(x >= 0., x < 1.)
        fx[idx] = 6 * x[idx] - 5 * Power(x[idx], 2)
        # if x >= 1 and x < 2
        idx = np.logical_and(x >= 1., x < 2.)
        fx[idx] = 7.5 - 9 * x[idx] + (5 * Power(x[idx], 2)) / 2.
        # if x >=2 and x < 3
        idx = np.logical_and(x >= 2., x < 3.)
        fx[idx] = -4.5 + 3 * x[idx] - Power(x[idx], 2) / 2.
        # if x < 0 and x > 3
        idx = np.logical_and(x < 0., x >= 3.)
        fx[idx] = 0
        return fx
