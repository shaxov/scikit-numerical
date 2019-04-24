import numpy as np
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt


class BaseBoundary(metaclass=ABCMeta):
    @abstractmethod
    def plot(self, save_path=None):
        """ Visualizes boundary. """

    @abstractmethod
    def _is_valid(self):
        """ Validate the data which gives for boundary initialization. """


class BaseIntegrableBoundary(metaclass=ABCMeta):
    @abstractmethod
    def _set_described_rect(self):
        """ Set n-dim rectangular bound which describes initial bound. """

    @abstractmethod
    def get_described_rect(self):
        """ Returns n-dim rectangular bound which describes initial bound. """


class BoundaryIterable(metaclass=ABCMeta):
    """ The class represents iterator interface for boundaries. """
    def __init__(self, bounds: list):
        self._bounds = bounds
        self.bounds_count = len(bounds)

    def __iter__(self):
        self._bound_id = 0
        return self

    def __next__(self):
        if self._bound_id == self.bounds_count:
            raise StopIteration
        bound = self._bounds[self._bound_id]
        self._bound_id += 1
        return bound

    def __getitem__(self, item):
        return self._bounds[item]

    def __repr__(self):
        return "<[" + ", ".join([bnd.__repr__() for bnd in self._bounds]) + "]>"


class LineBoundary(BaseBoundary):
    """ The class represents start and end of 1-dim line. """
    def __init__(self, start, end):
        self.start = np.float64(start)
        self.end = np.float64(end)

        if not self._is_valid():
            raise BoundarySetupException

    def _is_valid(self):
        return self.end > self.start

    def plot(self, save_path=None):
        plt.hlines(0.0, self.start, self.end, colors='black')
        plt.axvline(self.start, color='black', linestyle='--')
        plt.axvline(self.end, color='black', linestyle='--')
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def __repr__(self):
        return f"<{self.__class__.__name__}: start={self.start} end={self.end}>"


class PolarRhoLineBoundary(LineBoundary):
    """ The class represents start and end of 1-dim line
        in polar coordinates for Rho axis. """
    def __init__(self, start, end):
        # 'start' can be greater than 'end'
        super().__init__(start, end)

    def _is_valid(self):
        return (not np.allclose(self.start, self.end)
                and (self.start > 0.0 or np.allclose(self.start, 0.0))
                and not np.allclose(self.end, 0.0))


class PolarPhiLineBoundary(LineBoundary):
    """ The class represents start and end of 1-dim line
        in polar coordinates for Phi axis. """
    def __init__(self, start, end):
        # 'start' and 'end' can be in range from 0 to 2*Pi
        super().__init__(start, end)

    def _is_valid(self):
        return (self.start < self.end and not np.allclose(self.start, self.end)
                and (0.0 < self.start < 2*np.pi or np.allclose(self.start, 0.0) or np.allclose(self.start, 2*np.pi))
                and (0.0 < self.end < 2*np.pi or np.allclose(self.end, 0.0) or np.allclose(self.end, 2*np.pi)))


# ----------------------------------------------------------------------------------------------------------------------
# Boundary exceptions

class BoundarySetupException(Exception):
    """ The class defines exception which may arise
        during boundary initialization.
    """
    def __init__(self, message="bounds are not correct."):
        super().__init__(message)
