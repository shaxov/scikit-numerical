import numpy as np
from numpy import power as Power
from numerical import spline


class _Schoenberg1d:
    @staticmethod
    def deriv1(x: np.array) -> np.array:
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
    def deriv2(x: np.array) -> np.array:
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
    def deriv3(x: np.array) -> np.array:
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


def schoenberg1d(f, active_vars: tuple):
    _DERIVATIVES = {
        (active_vars[0], 1): _Schoenberg1d.deriv1,
        (active_vars[0], 2): _Schoenberg1d.deriv2,
        (active_vars[0], 3): _Schoenberg1d.deriv3,
    }
    f.deriv = lambda *args: _DERIVATIVES[args]
    return f


def schoenberg2d(f, active_vars: tuple):

    def _deriv10(x: np.array, y: np.array) -> np.array:
        return _Schoenberg1d.deriv1(x) * spline.schoenberg1d(y)

    def _deriv01(x: np.array, y: np.array) -> np.array:
        return spline.schoenberg1d(x) * _Schoenberg1d.deriv1(y)

    def _deriv11(x: np.array, y: np.array) -> np.array:
        return _Schoenberg1d.deriv1(x) * _Schoenberg1d.deriv1(y)

    def _deriv21(x: np.array, y: np.array) -> np.array:
        return _Schoenberg1d.deriv2(x) * _Schoenberg1d.deriv1(y)

    def _deriv12(x: np.array, y: np.array) -> np.array:
        return _Schoenberg1d.deriv1(x) * _Schoenberg1d.deriv2(y)

    _DERIVATIVES = {
        (active_vars[0], 1): _deriv10,
        (active_vars[1], 1): _deriv01,

        (active_vars[0], 1, active_vars[1], 1): _deriv11,
        (active_vars[1], 1, active_vars[0], 1): _deriv11,

        (active_vars[0], 2, active_vars[1], 1): _deriv21,
        (active_vars[1], 1, active_vars[0], 2): _deriv21,

        (active_vars[0], 1, active_vars[1], 2): _deriv12,
        (active_vars[1], 2, active_vars[0], 1): _deriv12,
    }

    f.deriv = lambda *args: _DERIVATIVES[args]
    return f