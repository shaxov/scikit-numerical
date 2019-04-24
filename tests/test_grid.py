import unittest
import numpy as np
from numerical.area import grid


class GridTest(unittest.TestCase):

    def test_uniform_grid(self):
        gd = grid.UniformGrid((0.0, 1.0), (0.05,))
        test_nodes = gd[0].nodes
        true_nodes = np.arange(0.0, 1.0, 0.05)
        true_nodes = np.append(true_nodes, 1.0)
        self.assertTrue(np.allclose(test_nodes, true_nodes))

        gd = grid.UniformGrid((-0.74, 1.25), (0.026,))
        test_nodes = gd[0].nodes
        true_nodes = np.arange(-0.74, 1.25, 0.026)
        true_nodes = np.append(true_nodes, 1.25)
        self.assertTrue(np.allclose(test_nodes, true_nodes))

