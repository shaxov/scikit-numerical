import unittest
import numpy as np

from numerical.integration import gauss
from numerical.area.grid import UniformGrid
from numerical.area import bounds


class GaussIntegrationTest(unittest.TestCase):
    def test_1d_integration(self):
        def f(x):
            return np.power(x[0], 2) + 2 * x[0] + 5

        grid = UniformGrid(bounds.cartesian.LineBoundary1D(-3.24, 9.24), [0.02])
        self.assertTrue(np.allclose(gauss.integrate(f, grid, 16, (64,)), [411.580]))

    def test_2d_integration(self):
        def f(x):
            return np.power(x[0], 2) + 2 * x[1] + 5

        grid = UniformGrid(bounds.cartesian.LineBoundary2D(0, 1, 0, 2 * np.pi), [0.02, np.pi / 8])
        self.assertTrue(np.allclose(gauss.integrate(f, grid, 16, (64, 64)), [72.9887]))

    def test_3d_integration(self):
        def f(x):
            return (7 * x[0] * x[2] - np.power(x[1], 2)) * x[2]

        grid = UniformGrid(bounds.cartesian.LineBoundary3D(-1, 1.5, -0.2, 0.5, 0, 1.8), [0.05, 0.05, 0.05])
        self.assertTrue(np.allclose(gauss.integrate(f, grid, 16, (32, 32, 32)), [5.77375]))

    def test_circle_integration(self):
        def f(x):
            return np.power(x[0], 2) + 2 * x[1] + 5
        grid = UniformGrid(bounds.polar.CircleBoundary(1.0), [0.1, np.pi/5])
        self.assertTrue(np.allclose(gauss.integrate(f, grid), [16.49336]))

    def test_half_circle_integration(self):
        def f(x):
            return np.power(x[0], 2) + 2 * x[1] + 5
        grid = UniformGrid(bounds.polar.HalfCircleBoundary(1.0), [0.1, np.pi/10])
        self.assertTrue(np.allclose(gauss.integrate(f, grid), [9.58001]))
