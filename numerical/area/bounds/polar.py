import numpy as np
from numerical.area.bounds import _base


class PolarCoordinates:
    COORDS_TYPE = "polar"


class CircleSegmentBoundary(_base.BoundaryIterable, PolarCoordinates):
    def __init__(self,
                 rho_start, rho_end,
                 phi_start, phi_end):
        super().__init__(
            [
                _base.PolarRhoLineBoundary(rho_start, rho_end),
                _base.PolarPhiLineBoundary(phi_start, phi_end),
            ]
        )
        self._set_described_rect()

    def _set_described_rect(self):
        self._described_rect = self._data

    def get_described_rect(self):
        return self._described_rect

    def plot(self, save_path=None):
        raise NotImplementedError


class HalfCircleBoundary(CircleSegmentBoundary):
    def __init__(self, radius):
        if radius < 0:
            raise _base.BoundarySetupException("Radius cen not be negative.")
        super().__init__(0.0, radius, 0.0, np.pi)

    def plot(self, save_path=None):
        raise NotImplementedError


class CircleBoundary(CircleSegmentBoundary):
    def __init__(self, radius):
        if radius < 0:
            raise _base.BoundarySetupException("Radius cen not be negative.")
        super().__init__(0.0, radius, 0.0, 2.*np.pi)

    def plot(self, save_path=None):
        raise NotImplementedError
