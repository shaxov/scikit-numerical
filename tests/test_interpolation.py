import unittest
import numpy as np
from numerical import interpolate


class InterpolationTest(unittest.TestCase):

    def test_interpolation(self):
        def fun1d(x):
            return np.power(x[0], 2) + 4

        meshgrid = [np.arange(0, 1.000001, 0.001)]
        values = fun1d(meshgrid)

        spline_fun = interpolate(values, meshgrid, 1)
        x = np.random.rand(1, 10)
        self.assertTrue(np.allclose(spline_fun(x), fun1d(x), atol=1e-1, rtol=1e-1))

        x = np.random.rand(1, 10, 10)
        self.assertTrue(np.allclose(spline_fun(x), fun1d(x), atol=1e-1, rtol=1e-1))

        x = np.random.rand(1, 10, 10, 10)
        self.assertTrue(np.allclose(spline_fun(x), fun1d(x), atol=1e-1, rtol=1e-1))

        def fun2d(x):
            return np.power(x[0], 3) * x[1] + 7 * x[1] + np.power(x[0], 0.5)

        grid1 = np.arange(0, 1.0001, 0.01)
        grid2 = np.arange(0, 1.0001, 0.01)
        meshgrid = np.meshgrid(grid1, grid2, indexing='ij')
        values = fun2d(meshgrid)

        spline_fun = interpolate(values, meshgrid, 1)

        x = np.random.rand(2, 5)
        self.assertTrue(np.allclose(spline_fun(x), fun2d(x), atol=1e-1, rtol=1e-1))

        x = np.random.rand(2, 5, 5)
        self.assertTrue(np.allclose(spline_fun(x), fun2d(x), atol=1e-1, rtol=1e-1))

        x = np.random.rand(2, 5, 5, 5)
        self.assertTrue(np.allclose(spline_fun(x), fun2d(x), atol=1e-1, rtol=1e-1))

        def fun3d(x):
            return (7 * x[0] * x[2] - np.power(x[1], 2)) * x[2]

        grid1 = np.arange(0, 1.0001, 0.01)
        grid2 = np.arange(0, 1.0001, 0.01)
        grid3 = np.arange(0, 1.0001, 0.01)

        meshgrid = np.meshgrid(grid1, grid2, grid3, indexing='ij')
        values = fun3d(meshgrid)

        spline_fun = interpolate(values, meshgrid, 1)

        x = np.random.rand(3, 5)
        self.assertTrue(np.allclose(spline_fun(x), fun3d(x), atol=1e-1, rtol=1e-1))

        x = np.random.rand(3, 5, 5)
        self.assertTrue(np.allclose(spline_fun(x), fun3d(x), atol=1e-1, rtol=1e-1))

        x = np.random.rand(3, 5, 5, 5)
        self.assertTrue(np.allclose(spline_fun(x), fun3d(x), atol=1e-1, rtol=1e-1))
