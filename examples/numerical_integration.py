""" Example of numerical integration using Gauss formula for 2-d function. """

import numpy as np
from numerical.integration import gauss
from numerical.area import bound
from numerical.area import grid


def f(x):
    return np.power(x[0], 2) + 2 * x[1] + 5


bd = bound.LineBoundary2D(0, 1, 0, 2 * np.pi)
grid = grid.UniformGrid(bd, [0.02, np.pi / 8])
gauss.integrate(f, grid)
