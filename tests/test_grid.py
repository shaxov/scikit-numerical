import unittest
import numpy as np
from numerical.area import bound
from numerical.area import grid


class GridTest(unittest.TestCase):

    def test_uniform_grid(self):
        bd = bound.LineBoundary1D(0.0, 1.0)
        gd = grid.UniformGrid(bd, step=[0.05])
        test_nodes = gd[0].get_nodes()
        true_nodes = np.arange(0.0, 1.0, 0.05)
        true_nodes = np.append(true_nodes, 1.0)
        self.assertTrue(np.allclose(test_nodes, true_nodes))

        true_sum = ((true_nodes[1:] + true_nodes[:-1]) / 2.0).reshape(-1, 1)
        test_sum = gd[0].sum
        self.assertTrue(np.allclose(test_sum, true_sum))

        true_diff = (true_nodes[1:] - true_nodes[:-1]) / 2.0
        test_diff = gd[0].diff
        self.assertTrue(np.allclose(test_diff, true_diff))

        true_diff_t = true_diff.reshape(-1, 1)
        test_diff_t = gd[0].diff_t
        self.assertTrue(np.allclose(test_diff_t, true_diff_t))

        bd = bound.LineBoundary1D(-0.74, 1.25)
        gd = grid.UniformGrid(bd, step=[0.026])
        test_nodes = gd[0].get_nodes()
        true_nodes = np.arange(-0.74, 1.25, 0.026)
        true_nodes = np.append(true_nodes, 1.25)
        self.assertTrue(np.allclose(test_nodes, true_nodes))

        true_sum = ((true_nodes[1:] + true_nodes[:-1]) / 2.0).reshape(-1, 1)
        test_sum = gd[0].sum
        self.assertTrue(np.allclose(test_sum, true_sum))

        true_diff = (true_nodes[1:] - true_nodes[:-1]) / 2.0
        test_diff = gd[0].diff
        self.assertTrue(np.allclose(test_diff, true_diff))

        true_diff_t = true_diff.reshape(-1, 1)
        test_diff_t = gd[0].diff_t
        self.assertTrue(np.allclose(test_diff_t, true_diff_t))
