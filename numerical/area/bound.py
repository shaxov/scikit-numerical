import numpy as np
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt


class _BaseBoundary(metaclass=ABCMeta):
    @abstractmethod
    def plot(self, save_path=None):
        """ Visualizes boundary. """

    def _set_described_rect(self):
        """ Set n-dim rectangular bound which describes initial bound. """


class _BoundaryIterable(metaclass=ABCMeta):
    """ The class represents iterator interface for boundaries. """
    def __init__(self, data: list):
        self._data = data
        self._bounds_count = len(data)

    def __iter__(self):
        self._bound_id = -1
        return self

    def __next__(self):
        if self._bound_id == self._bounds_count - 1:
            raise StopIteration
        self._bound_id += 1
        return self._data[self._bound_id]

    def __getitem__(self, item):
        return self._data[item]

    def __repr__(self):
        return "<[" + ", ".join([bnd.__repr__() for bnd in self._data]) + "]>"


class _LineBoundary(_BaseBoundary):
    """ The class represents start and end of 1-dim line. """
    def __init__(self, start, end):
        if end < start or np.allclose(start, end):
            raise BoundarySetupException

        self.start = np.float64(start)
        self.end = np.float64(end)

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


class LineBoundary1D(_BoundaryIterable):
    def __init__(self, x_start, x_end):
        super().__init__(
            [
                _LineBoundary(x_start, x_end),
            ]
        )
        self._set_described_rect()

    def _set_described_rect(self):
        self._described_rect = self._data

    def plot(self, save_path=None):
        self._data[0].plot(save_path)


class LineBoundary2D(_BoundaryIterable):
    def __init__(self,
                 x_start, x_end,
                 y_start, y_end):
        super().__init__(
            [
                _LineBoundary(x_start, x_end),
                _LineBoundary(y_start, y_end),
            ]
        )
        self._set_described_rect()

    def _set_described_rect(self):
        self._described_rect = self._data

    def plot(self, save_path=None):
        x_start, x_end = self._data[0].start, self._data[0].end
        y_start, y_end = self._data[1].start, self._data[1].end
        x_start2, x_end2 = self._described_rect[0].start, self._described_rect[0].end
        y_start2, y_end2 = self._described_rect[1].start, self._described_rect[1].end

        plt.hlines([y_start, y_end], x_start, x_end, colors='black', linestyle='-', linewidth=3)
        plt.hlines([y_start2, y_end2], x_start2, x_end2, colors='red', linestyle='--', linewidth=1)
        plt.vlines([x_start, x_end], y_start, y_end, colors='black', linestyle='-', linewidth=3)
        plt.vlines([x_start2, x_end2], y_start2, y_end2, colors='red', linestyle='--', linewidth=1)

        plt.legend(('Defined boundary', 'Described rect'),
                   loc='upper right')

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


class LineBoundary3D(_BoundaryIterable):
    def __init__(self,
                 x_start, x_end,
                 y_start, y_end,
                 z_start, z_end):
        super().__init__(
            [
                _LineBoundary(x_start, x_end),
                _LineBoundary(y_start, y_end),
                _LineBoundary(z_start, z_end),
            ]
        )
        self._set_described_rect()

    def _set_described_rect(self):
        self._described_rect = self._data

    def plot(self, *args):
        raise NotImplementedError


class BoundarySetupException(Exception):
    """ The class defines exception which may arise
        during boundary initialization.
    """
    def __init__(self, message="bounds are not correct."):
        super().__init__(message)
