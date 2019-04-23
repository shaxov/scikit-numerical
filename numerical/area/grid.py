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
        self._grid_id = -1
        return self

    def __next__(self):
        if self._grid_id == self.dim - 1:
            raise StopIteration
        self._grid_id += 1
        return self._data[self._grid_id]

    def __getitem__(self, item):
        return self._data[item]

    def __repr__(self):
        return "<[" + ", ".join([grd.__repr__() for grd in self._data]) + "]>"


class _UniformGrid(_BaseGrid):
    """ Class defines uniform grid object.

    'Uniform' means that distance between nearest nodes is the same
    for each node and defines as 'step'.
    """
    def __init__(self, bound, step):
        self._bound = bound
        self._step = step

        self._build_nodes()
        self._build_integration_meta()

    def _build_nodes(self):
        self._nodes = np.arange(self._bound.start, self._bound.end, self._step)
        self._nodes = np.append(self._nodes, self._bound.end)

    def _build_integration_meta(self):
        self.sum = ((self._nodes[1:] + self._nodes[:-1]) / 2.0).reshape(-1, 1)
        self.diff = (self._nodes[1:] - self._nodes[:-1]) / 2.0
        self.diff_t = self.diff.reshape(-1, 1)
        self.nodes_count = len(self._nodes)

    def get_nodes(self):
        return self._nodes

    def __repr__(self):
        return f"<{self.__class__.__name__}: " \
            f"bound={str(self._bound)}, " \
            f"step={str(self._step)}, " \
            f"nodes_count={len(self._nodes)}>"


class UniformGrid(_GridIterable):
    def __init__(self, bound: "area boundary", step: list):
        if bound.bounds_count != len(step):
            raise ValueError("Boundary dimension and steps count don't match.")
        super().__init__(
            [_UniformGrid(bd, st) for bd, st in zip(bound.get_described_rect(), step)]
        )
        self.coords_type = bound.COORDS_TYPE
