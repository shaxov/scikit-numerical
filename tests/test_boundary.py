import unittest
import numpy as np
from numerical.area import bounds


class BoundaryTest(unittest.TestCase):

    def test_line_boundary_1d_init(self):
        bd = bounds.cartesian.LineBoundary1D(0.0, 1.0)
        self.assertAlmostEqual(0.0, bd[0].start)
        self.assertAlmostEqual(1.0, bd[0].end)

    def test_line_boundary_2d_init(self):
        bd = bounds.cartesian.LineBoundary2D(0.0, 1.0, 0.5, 1.5)
        self.assertAlmostEqual(0.0, bd[0].start)
        self.assertAlmostEqual(1.0, bd[0].end)
        self.assertAlmostEqual(0.5, bd[1].start)
        self.assertAlmostEqual(1.5, bd[1].end)

    def test_line_boundary_3d_init(self):
        bd = bounds.cartesian.LineBoundary3D(0.0, 1.0, 0.5, 1.5, -0.25, 1.5)
        self.assertAlmostEqual(0.0, bd[0].start)
        self.assertAlmostEqual(1.0, bd[0].end)
        self.assertAlmostEqual(0.5, bd[1].start)
        self.assertAlmostEqual(1.5, bd[1].end)
        self.assertAlmostEqual(-0.25, bd[2].start)
        self.assertAlmostEqual(1.5, bd[2].end)

    def test_line_boundary_1d_exception(self):
        self.assertRaises(bounds.BoundarySetupException, bounds.cartesian.LineBoundary1D, 10.0, 1.0)

    def test_line_boundary_2d_exception(self):
        self.assertRaises(bounds.BoundarySetupException, bounds.cartesian.LineBoundary2D, 0.0, 1.0, 0.5, 0.5000001)

    def test_line_boundary_3d_exception(self):
        self.assertRaises(bounds.BoundarySetupException, bounds.cartesian.LineBoundary3D, 0.0, 1.0, 0.5, 1.5, -0.25, -1.5)

    def test_polar_rho_phi_boundary(self):
        from numerical.area.bounds._base import PolarPhiLineBoundary, PolarRhoLineBoundary
        bd_rho = PolarRhoLineBoundary(0.0, 1.0)
        bd_phi = PolarPhiLineBoundary(0.0, 2.0*np.pi)

        self.assertAlmostEqual(bd_rho.start, 0.0)
        self.assertAlmostEqual(bd_rho.end, 1.0)

        self.assertAlmostEqual(bd_phi.start, 0.0)
        self.assertAlmostEqual(bd_phi.end, 2.0*np.pi)

    def test_polar_phi_exception(self):
        from numerical.area.bounds._base import PolarPhiLineBoundary
        self.assertRaises(bounds.BoundarySetupException, PolarPhiLineBoundary, 0.0, 2.0 * np.pi + 1e-4)
        self.assertRaises(bounds.BoundarySetupException, PolarPhiLineBoundary, 0.0 - 1e-4, 2.0 * np.pi)
        self.assertRaises(bounds.BoundarySetupException, PolarPhiLineBoundary, 2.0 * np.pi, 0.0)

    def test_polar_rho_exception(self):
        from numerical.area.bounds._base import PolarRhoLineBoundary
        self.assertRaises(bounds.BoundarySetupException, PolarRhoLineBoundary, -0.01, 2.0 * np.pi)

    def test_circle_boundaries(self):
        bd_cs = bounds.polar.CircleSegmentBoundary(0.0, 1.5, 0.0, 0.5 * np.pi)
        bd_c = bounds.polar.CircleBoundary(5.0)
        bd_hc = bounds.polar.HalfCircleBoundary(5.0)

        self.assertAlmostEqual(bd_cs[0].start, 0.0)
        self.assertAlmostEqual(bd_cs[0].end, 1.5)
        self.assertAlmostEqual(bd_cs[1].start, 0.0)
        self.assertAlmostEqual(bd_cs[1].end, 0.5 * np.pi)

        self.assertAlmostEqual(bd_c[0].start, 0.0)
        self.assertAlmostEqual(bd_c[0].end, 5.0)
        self.assertAlmostEqual(bd_c[1].start, 0.0)
        self.assertAlmostEqual(bd_c[1].end, 2. * np.pi)

        self.assertAlmostEqual(bd_hc[0].start, 0.0)
        self.assertAlmostEqual(bd_hc[0].end, 5.0)
        self.assertAlmostEqual(bd_hc[1].start, 0.0)
        self.assertAlmostEqual(bd_hc[1].end, np.pi)

    def test_circle_boundary_exception(self):
        self.assertRaises(bounds.BoundarySetupException, bounds.polar.CircleBoundary, -5.0)

    def test_half_circle_boundary_exception(self):
        self.assertRaises(bounds.BoundarySetupException, bounds.polar.HalfCircleBoundary, -5.0)




