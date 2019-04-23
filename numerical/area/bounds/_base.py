import numpy as np
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt


class BaseBoundary(metaclass=ABCMeta):
    @abstractmethod
    def plot(self, save_path=None):
        """ Visualizes boundary. """

    def _set_described_rect(self):
        """ Set n-dim rectangular bound which describes initial bound. """

    def _is_boundary_valid(self, *args, **kwargs):
        """ Validate the data which gives for boundary initialization. """


class BoundaryIterable(metaclass=ABCMeta):
    """ The class represents iterator interface for boundaries. """
    def __init__(self, data: list):
        self._data = data
        self._bounds_count = len(data)

    def __iter__(self):
        self._bound_id = 0
        return self

    def __next__(self):
        if self._bound_id == self._bounds_count:
            raise StopIteration
        self._bound_id += 1
        return self._data[self._bound_id - 1]

    def __getitem__(self, item):
        return self._data[item]

    def __repr__(self):
        return "<[" + ", ".join([bnd.__repr__() for bnd in self._data]) + "]>"


class LineBoundary(BaseBoundary):
    """ The class represents start and end of 1-dim line. """
    def __init__(self, start, end):
        if not self._is_boundary_valid(start, end):
            raise BoundarySetupException

        self.start = np.float64(start)
        self.end = np.float64(end)

    def _is_boundary_valid(self, start, end):
        return end > start and not np.allclose(start, end)

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

    def _is_boundary_valid(self, start, end):
        return not np.allclose(start, end) and (start > 0.0 or np.allclose(start, 0.0)) and not np.allclose(end, 0.0)


class PolarPhiLineBoundary(LineBoundary):
    """ The class represents start and end of 1-dim line
        in polar coordinates for Phi axis. """
    def __init__(self, start, end):
        # 'start' and 'end' can be in range from 0 to 2*Pi
        super().__init__(start, end)

    def _is_boundary_valid(self, start, end):
        return (start < end and not np.allclose(start, end)
                and (0.0 < start < 2*np.pi or np.allclose(start, 0.0) or np.allclose(start, 2*np.pi))
                and (0.0 < end < 2*np.pi or np.allclose(end, 0.0) or np.allclose(end, 2*np.pi)))


# ----------------------------------------------------------------------------------------------------------------------
# Boundary exceptions

class BoundarySetupException(Exception):
    """ The class defines exception which may arise
        during boundary initialization.
    """
    def __init__(self, message="bounds are not correct."):
        super().__init__(message)
