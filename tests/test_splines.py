import unittest
import numpy as np
from numerical import splines


class SplinesTest(unittest.TestCase):

    def test_linear_spline(self):
        var = np.array([0.0, 0.256, -0.756, 1.5, 1.0])
        self.assertTrue(np.allclose(splines.linear(var),
                                    np.array([1.0, 0.744, 0.244, 0.0, 0.0])))

    def test_shenberg_spline(self):
        var = np.array([-1.52, 0.001, -0.001, 0.0, 2.356, 1.254, 3.260, 3.0])
        self.assertTrue(np.allclose(splines.shenberg(var),
                        np.array([0, 0.5499995, 0, 0.55, 0.0009231, 0.1236675, 0., 0.])))

