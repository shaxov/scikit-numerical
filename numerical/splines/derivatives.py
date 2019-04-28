import numpy as np
from numpy import power as Power


def schoenberg_spline_derivatives(f):
    def deriv(x, order):
        if isinstance(order, tuple):
            if len(order) > 1:
                raise ValueError("Function is 1-dimensional. Mixed derivative does not exist.")
            if len(order) == 0:
                raise ValueError("Please, specify the order of derivative.")
            order = order[0]
        if order == 1:
            return _deriv1(x)
        elif order == 2:
            return _deriv2(x)
        elif order == 3:
            return _deriv3(x)
        else:
            raise NotImplementedError(f"Derivative of order {order} for Schoenberg splines is not implemented. "
                                      f"Max order of available derivative is 3.")

    def _deriv1(x: np.array) -> np.array:
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

    def _deriv2(x: np.array) -> np.array:
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

    def _deriv3(x: np.array) -> np.array:
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

    f.deriv = deriv
    return f
