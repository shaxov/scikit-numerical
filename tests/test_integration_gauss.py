import unittest
import numpy as np

from numerical.integration import gauss
from numerical.area.grid import Grid
from numerical.area.boundary import Boundary


class GaussIntegrationTest(unittest.TestCase):
    def test_1d_integration(self):
        def f(x):
            return np.power(x[0], 2) + 2 * x[0] + 5

        grid = Grid([Boundary(-3.24, 9.24)], [0.02])
        self.assertTrue(np.allclose(gauss.integrate(f, grid, 16, (64,)), [411.580]))

    def test_2d_integration(self):
        def f(x):
            return np.power(x[0], 2) + 2 * x[1] + 5

        boundary_x = Boundary(0, 1)
        boundary_y = Boundary(0, 2 * np.pi)
        grid = Grid([boundary_x, boundary_y], [0.02, np.pi / 8])
        self.assertTrue(np.allclose(gauss.integrate(f, grid, 16, (64, 64)), [72.9887]))

    def test_3d_integration(self):
        def f(x):
            return (7 * x[0] * x[2] - np.power(x[1], 2)) * x[2]
        boundary_x = Boundary(-1, 1.5)
        boundary_y = Boundary(-0.2, 0.5)
        boundary_z = Boundary(0, 1.8)
        grid = Grid([boundary_x, boundary_y, boundary_z], [0.05, 0.05, 0.05])
        self.assertTrue(np.allclose(gauss.integrate(f, grid, 16, (32, 32, 32)), [5.77375]))
