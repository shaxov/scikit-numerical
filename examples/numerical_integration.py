""" Example of numerical integration using Gauss formula for 2-d function. """

import numpy as np
from numerical.integration import gauss
from numerical.area.boundary import Boundary
from numerical.area.grid import Grid


def f(x):
    return np.power(x[0], 2) + 2 * x[1] + 5


boundary_x = Boundary(0, 1)
boundary_y = Boundary(0, 2 * np.pi)
grid = Grid([boundary_x, boundary_y], [0.02, np.pi / 8])
gauss.integrate(f, grid)
