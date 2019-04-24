import numpy as np
from abc import ABCMeta, abstractmethod


class _BaseGrid(metaclass=ABCMeta):
    """ Base class for grid kinds. """
    @abstractmethod
    def _build_nodes(self):
        """ Builds grid on predefined boundary. """

    @abstractmethod
    def __repr__(self):
        """ Defines str representation of grid object. """


class _GridIterable(metaclass=ABCMeta):
    """ The class represents iterator interface for grid. """

    def __init__(self, data: list):
        self._data = data
        self.dim = len(data)

    def __iter__(self):
        self._grid_id = 0
        return self

    def __next__(self):
        if self._grid_id == self.dim:
            raise StopIteration
        grid = self._data[self._grid_id]
        self._grid_id += 1
        return grid

    def __getitem__(self, item):
        return self._data[item]

    def __repr__(self):
        return "<[" + ", ".join([grd.__repr__() for grd in self._data]) + "]>"


class _UniformGrid(_BaseGrid):
    """ Class defines uniform grid object.

    'Uniform' means that distance between nearest nodes is the same
    for each node and defines as 'step'.
    """
    def __init__(self, start, end, step):
        self.start = start
        self.end = end
        self.step = step
        self._build_nodes()

    def _build_nodes(self):
        self.nodes = np.arange(self.start, self.end, self.step)
        self.nodes = np.append(self.nodes, self.end)

    def get_nodes(self):
        return self.nodes

    def __repr__(self):
        return f"<{self.__class__.__name__}: " \
            f"start={self.start}, " \
            f"end={self.end}, " \
            f"step={self.step}, " \
            f"nodes_count={len(self.nodes)}>"


class UniformGrid(_GridIterable):
    def __init__(self, bounds: tuple, steps: tuple):
        try:
            bounds = np.array(bounds).reshape(-1, 2)
        except ValueError:
            raise ValueError("Number of bounds must be even. (2, 4 or 6).")

        if not steps:
            steps = (0.05,) * bounds.shape[0]
        steps = np.array(steps).reshape(-1, 1)

        if len(bounds) != len(steps):
            raise ValueError(f"Boundary dimension and steps count don't match. {len(bounds)} != {len(steps)}")

        super().__init__(
            [_UniformGrid(*bd, st) for bd, st in zip(bounds, steps)]
        )
