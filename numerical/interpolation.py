import numpy as np

from numerical import splines
from numerical.utils.interpolation import repeat_args


def interpolate(values, grid, batch_size=16):
    """ Builds function which is an interpolation function on nodes with computer values in these nodes.

    Args:
        values: list of function values in grid nodes.
        grid: points where function was calculated used np.meshgrid function with parameter 'indexing='ij''.
        batch_size: int, batch size for interpolation process.

    Returns:
        interpolated function.
    """
    # linear interpolation will be used as a basis function
    bfunc = splines.linear

    nodes_range = [(g.min(), g.max()) for g in grid]
    nodes_count = grid[0].shape
    nodes_dim = len(grid)
    shift_indexes = [np.arange(0, dim, dtype=np.float64) for dim in nodes_count]
    values = values.ravel()

    def _interpolated(x):
        """ Interpolated function.

        Args:
            x: numpy.ndarray

        Returns:
            numpy.ndarray
        """

        if x.shape[-1] != 1:
            x = np.expand_dims(x, axis=-1)

        result = []
        batch_position = 0
        _interpolation_loop(result, x, values, bfunc, nodes_range, nodes_count,
                            nodes_dim, shift_indexes, batch_position, batch_size)
        return np.concatenate(result)

    return _interpolated


def _interpolation_loop(result, x, values, bfunc, nodes_range, nodes_count,
                        nodes_dim, shift_indexes, batch_position, batch_size):
    while batch_position < x.shape[1]:
        args = []
        for i in range(nodes_dim):
            axis_batch = x[i][batch_position:batch_position + batch_size]
            arg = ((nodes_count[i] - 1) * (axis_batch - nodes_range[i][0]) /
                   (nodes_range[i][1] - nodes_range[i][0])) - shift_indexes[i]
            args.append(arg)

        rep_args = repeat_args(args, nodes_count)
        spline_val = np.prod(np.array([bfunc(arg) for arg in rep_args]), axis=0)
        if len(spline_val.shape) == 1:
            spline_val = spline_val.reshape(1, -1)
        result.append(np.dot(spline_val, values))
        batch_position += batch_size
