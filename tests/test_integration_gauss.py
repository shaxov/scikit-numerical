import unittest
import numpy as np

from numerical.integration import gauss
from numerical.area.grid import UniformGrid


class GaussIntegrationTest(unittest.TestCase):
    def test_1d_integration(self):
        def f(x):
            return np.power(x[0], 2) + 2 * x[0] + 5

        grid = UniformGrid((-3.24, 9.24), (0.02,))
        self.assertTrue(np.allclose(gauss.integrate(f, ndgrid=grid, roots_count=16, batch_size=(64,)), [411.580]))
        self.assertTrue(np.allclose(gauss.integrate(f, -3.24, 9.24), [411.580]))

    def test_2d_integration(self):
        def f(x):
            return np.power(x[0], 2) + 2 * x[1] + 5

        grid = UniformGrid((0, 1, 0, 2 * np.pi), (0.02, np.pi / 8))
        self.assertTrue(np.allclose(gauss.integrate(f, ndgrid=grid, roots_count=16, batch_size=(64, 64)), [72.9887]))
        self.assertTrue(np.allclose(gauss.integrate(f, 0, 1, 0, 2 * np.pi, steps=(0.02, np.pi / 8)), [72.9887]))

    def test_3d_integration(self):
        def f(x):
            return (7 * x[0] * x[2] - np.power(x[1], 2)) * x[2]

        grid = UniformGrid((-1, 1.5, -0.2, 0.5, 0, 1.8), (0.05, 0.05, 0.05))
        self.assertTrue(np.allclose(gauss.integrate(f, ndgrid=grid, roots_count=16, batch_size=(32, 32, 32)), [5.77375]))
        self.assertTrue(
            np.allclose(
                gauss.integrate(f, -1, 1.5, -0.2, 0.5, 0, 1.8, roots_count=16, batch_size=(32, 32, 32)), [5.77375])
        )

    def test_circle_integration(self):
        def f(x):
            return np.power(x[0], 2) + 2 * x[1] + 5
        grid = UniformGrid((0, 1, 0, 2 * np.pi), (0.1, np.pi/5))
        self.assertTrue(np.allclose(gauss.integrate(f, ndgrid=grid, coords_type="polar"), [16.49336]))
        self.assertTrue(np.allclose(gauss.integrate(f, 0, 1, 0, 2 * np.pi, steps=(0.1, np.pi/5), coords_type="polar"),
                                    [16.49336]))

    def test_half_circle_integration(self):
        def f(x):
            return np.power(x[0], 2) + 2 * x[1] + 5
        grid = UniformGrid((0, 1, 0, np.pi), (0.1, np.pi/10))
        self.assertTrue(np.allclose(gauss.integrate(f, ndgrid=grid, coords_type="polar"), [9.58001]))
        self.assertTrue(np.allclose(gauss.integrate(f, 0, 1, 0, np.pi, steps=(0.1, np.pi/10), coords_type="polar"),
                                    [9.58001]))

