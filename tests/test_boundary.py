import unittest
from numerical.area import bound


class BoundaryTest(unittest.TestCase):

    def test_line_boundary_1d_init(self):
        bd = bound.LineBoundary1D(0.0, 1.0)
        self.assertAlmostEqual(0.0, bd[0].start)
        self.assertAlmostEqual(1.0, bd[0].end)

    def test_line_boundary_2d_init(self):
        bd = bound.LineBoundary2D(0.0, 1.0, 0.5, 1.5)
        self.assertAlmostEqual(0.0, bd[0].start)
        self.assertAlmostEqual(1.0, bd[0].end)
        self.assertAlmostEqual(0.5, bd[1].start)
        self.assertAlmostEqual(1.5, bd[1].end)

    def test_line_boundary_3d_init(self):
        bd = bound.LineBoundary3D(0.0, 1.0, 0.5, 1.5, -0.25, 1.5)
        self.assertAlmostEqual(0.0, bd[0].start)
        self.assertAlmostEqual(1.0, bd[0].end)
        self.assertAlmostEqual(0.5, bd[1].start)
        self.assertAlmostEqual(1.5, bd[1].end)
        self.assertAlmostEqual(-0.25, bd[2].start)
        self.assertAlmostEqual(1.5, bd[2].end)

    def test_line_boundary_1d_exception(self):
        self.assertRaises(bound.BoundarySetupException, bound.LineBoundary1D, 10.0, 1.0)

    def test_line_boundary_2d_exception(self):
        self.assertRaises(bound.BoundarySetupException, bound.LineBoundary2D, 0.0, 1.0, 0.5, 0.5000001)

    def test_line_boundary_3d_exception(self):
        self.assertRaises(bound.BoundarySetupException, bound.LineBoundary3D, 0.0, 1.0, 0.5, 1.5, -0.25, -1.5)

