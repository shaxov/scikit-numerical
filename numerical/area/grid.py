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

    def __repr__(self):
        return f"<{self.__class__.__name__}: " \
            f"start={self.start}, " \
            f"end={self.end}, " \
            f"step={self.step}, " \
            f"nodes_count={len(self.nodes)}>"


class UniformGrid:
    def __init__(self, bounds: tuple, steps: tuple):
        try:
            bounds = np.array(bounds).reshape(-1, 2)
        except ValueError:
            raise ValueError("Number of bounds must be even. (2, 4 or 6).")

        if len(bounds) > 6:
            raise ValueError(f"Max number of bounds is 6. {len(bounds)} > 6")

        if not steps:
            steps = (0.05,) * bounds.shape[0]
        steps = np.array(steps).reshape(-1, 1)

        if len(bounds) != len(steps):
            raise ValueError(f"Boundary dimension and steps count don't match. {len(bounds)} != {len(steps)}")

        self._grids = [_UniformGrid(*bd, st) for bd, st in zip(bounds, steps)]
        self.dim = len(self._grids)

    def __iter__(self):
        self._grid_id = 0
        return self

    def __next__(self):
        if self._grid_id == self.dim:
            raise StopIteration
        grid = self._grids[self._grid_id]
        self._grid_id += 1
        return grid

    def __getitem__(self, item):
        return self._grids[item]

    def __repr__(self):
        return "<[" + ", ".join([grd.__repr__() for grd in self._grids]) + "]>"
