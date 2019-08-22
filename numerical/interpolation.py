import numpy as np

import _interpolation


def interpolate(values, grid):
    """ Builds function which is an interpolation function on nodes with computer values in these nodes.

    Args:
        values: list of function values in grid nodes.
        grid: points where function was calculated used np.meshgrid function with parameter 'indexing='ij''.

    Returns:
        interpolated function.
    """

    nodes_range = tuple(np.ravel([(g.min(), g.max()) for g in grid]))
    nodes_count = grid[0].shape
    nodes_dim = len(grid)

    values = values.reshape(nodes_count)

    if nodes_dim == 1:
        def _interpolated(x):
            """ Interpolated function.

            Args:
                x: numpy.ndarray

            Returns:
                numpy.ndarray
            """
            return _interpolation.line_interpolate_1d(x[0], values, nodes_range)
    elif nodes_dim == 2:
        def _interpolated(x):
            """ Interpolated function.

            Args:
                x: numpy.ndarray

            Returns:
                numpy.ndarray
            """
            return _interpolation.line_interpolate_2d(x[0], x[1], values, nodes_range)
    elif nodes_dim == 3:
        def _interpolated(x):
            """ Interpolated function.

            Args:
                x: numpy.ndarray

            Returns:
                numpy.ndarray
            """
            return _interpolation.line_interpolate_3d(x[0], x[1], x[2], values, nodes_range)
    else:
        raise NotImplementedError

    return _interpolated
