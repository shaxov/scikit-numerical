import unittest
import numpy as np
from numerical import splines
from numerical.derivative import diff


class SplinesTest(unittest.TestCase):

    def test_linear_spline(self):
        var = np.array([0.0, 0.256, -0.756, 1.5, 1.0])
        self.assertTrue(np.allclose(splines.linear(var),
                                    np.array([1.0, 0.744, 0.244, 0.0, 0.0])))

    def test_shenberg_spline(self):
        var = np.array([-1.52, 0.001, -0.001, 0.0, 2.356, 1.254, 3.260, 3.0])
        self.assertTrue(np.allclose(splines.schoenberg(var),
                        np.array([0, 0.5499995, 0, 0.55, 0.0009231, 0.1236675, 0., 0.])))
        d1 = diff(splines.schoenberg, "x", 1)
        self.assertTrue(np.allclose(d1(var),
                                    [0., -0.001, 0., 0., -0.00716691, -0.30979956, 0., 0.]))
        d2 = diff(splines.schoenberg, "x", 2)
        self.assertTrue(np.allclose(d2(var),
                                    [0., -0.999997, 0., -1., 0.044515, 0.47195722, 0., 0.]))
        d3 = diff(splines.schoenberg, "x", 3)
        self.assertTrue(np.allclose(d3(var),
                                    [0., 0.005995, 0., 0., -0.207368, 0.14529, 0., 0.]))
