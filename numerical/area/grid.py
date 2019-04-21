import numpy as np
import itertools
from numerical.area.boundary import Boundary
from numerical.utils.coord_transform import polar2cartesian
from typing import List

EPS = 1e-8


class _grid:
    """ 1-dim grid """
    def __init__(self, boundary, step, polarized=False):
        self._cartesian_data = None
        self._polarized_data = None

        self._data = _grid._build(boundary, step)
        if polarized:
            self._polarized_data = self._data
        else:
            self._cartesian_data = self._data

    @staticmethod
    def _build(boundary, step):
        data = {}
        data = _grid._build_mesh_data(data, boundary, step)
        data = _grid._build_meta_data(data)
        return data

    @staticmethod
    def _build_mesh_data(data, boundary, step):
        """ Creates n-dim mesh data """
        data['boundary'] = boundary
        data['step'] = step
        data['n_roots'] = int((boundary.end - boundary.start) / step + EPS)
        data['mesh'] = boundary.start + np.array([i * step for i in range(data['n_roots'] + 1)],
                                                 dtype=np.float64)
        return data

    @staticmethod
    def _build_meta_data(data):
        """ Creates n-dim metadata for integration and interpolation. """
        data = _grid._add_integration_meta(data)
        data = _grid._add_interpolation_meta(data)
        return data

    @staticmethod
    def _add_integration_meta(data):
        data['sum'] = ((data['mesh'][1:] + data['mesh'][:-1]) / 2.0).reshape(-1, 1)
        data['diff'] = (data['mesh'][1:] - data['mesh'][:-1]) / 2.0
        data['diff_t'] = data['diff'].reshape(-1, 1)
        return data

    @staticmethod
    def _add_interpolation_meta(data):
        data['b1spline_iter'] = np.arange(0, data['n_roots'] + 1, dtype=np.float64)
        return data

    def build_cartesian(self, boundary, step):
        self._cartesian_data = _grid._build(boundary, step)

    def build_polarized(self, boundary, step):
        self._polarized_data = _grid._build(boundary, step)

    def use_cartesian(self):
        """ Switch to cartesian data. """
        self._data = self._cartesian_data

    def use_polarized(self):
        """ Switch to polar data. """
        self._data = self._polarized_data

    def __repr__(self):
        return f"<Grid:" \
               f" start={self._data['mesh'][0]}," \
               f" end={self._data['mesh'][-1]}," \
               f" step={self._data['step']}," \
               f" roots={self._data['n_roots']}," \
               f" polarized={str(bool(self._polarized_data))}>"

    def __getitem__(self, item):
        return self._data[item]


class Grid:
    """ Generalization of Grid for N-dim. """

    def __init__(self, boundaries: List[Boundary], steps: List[float], polarized=False):
        self.boundaries = boundaries
        self.steps = np.array(steps, dtype=np.float64)
        self.polarized = polarized
        self.dim = len(self.steps)

        self._cartesian_data = None
        self._polarized_data = None

        self._data = {}
        self._build()

    def _build_grids(self):
        self._grids = []
        for grid_id, (boundary, step) in enumerate(zip(self.boundaries, self.steps)):
            grid = _grid(boundary, step, self.polarized)
            self._grids.append(grid)

    def _count_roots(self, data):
        data['n_roots'] = np.prod([grid['n_roots'] for grid in self._grids])
        return data

    def _build_mesh(self, data):
        data['mesh'] = np.array(list(itertools.product(*[g['mesh'] for g in self._grids])), dtype=np.float64)
        return data

    def _set_cartesian_mesh(self):
        self._data['mesh'] = polar2cartesian(self._data['mesh'].T).T

    def _build(self):
        self._build_grids()
        self._data = self._count_roots(self._data)
        self._data = self._build_mesh(self._data)

        if self.polarized:
            self._polarized_data = self._data
            self._set_cartesian_mesh()
        else:
            self._cartesian_data = self._data

    def __repr__(self):
        return '[' + ",\n ".join([g.__repr__() for g in self._grids]) + ']'

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._data[item]
        else:
            return self._grids[item]

    def __iter__(self):
        self._n = 0
        return self

    def __next__(self):
        self._n += 1
        if self._n - 1 >= self.dim:
            raise StopIteration
        return self._grids[self._n - 1]
