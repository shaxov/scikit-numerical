import numpy as np

from numerical import splines
from numerical.utils.interpolation import repeat_args


def interpolate(values, meshgrid, batch_size=16, bfunc=None):
    """ Builds function which is an interpolation function on nodes with computer values in these nodes.

    Args:
        values: list of function values in grid nodes.
        meshgrid: points where function was calculated.
        batch_size: int, batch size for interpolation process.
        bfunc: basis function which is used to interpolate spaces between nodes.

    Returns:
        interpolated function.
    """
    # if basis function is not set, linear interpolation will be used
    if bfunc is None:
        bfunc = splines.linear

    nodes_range = [(grid.min(), grid.max()) for grid in meshgrid]
    nodes_count = meshgrid[0].shape
    nodes_dim = len(meshgrid)
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


if __name__ == '__main__':
    def fun(x):
        return np.power(x[0], 2) + 4

    meshgrid = [np.arange(0, 1.000001, 0.001)]
    values = fun(meshgrid)

    spline_fun = interpolate(values, meshgrid)
    x = np.random.rand(1, 10)
    assert np.allclose(spline_fun(x), fun(x), atol=1e-2, rtol=1e-2)

    x = np.random.rand(1, 10, 10)
    assert np.allclose(spline_fun(x), fun(x), atol=1e-2, rtol=1e-2)

    x = np.random.rand(1, 10, 10, 10)
    assert np.allclose(spline_fun(x), fun(x), atol=1e-2, rtol=1e-2)


    def fun(x):
        return np.power(x[0], 3) * x[1] + 7 * x[1] + np.power(x[0], 0.5)


    grid1 = np.arange(0, 1.0001, 0.01)
    grid2 = np.arange(0, 1.0001, 0.01)
    meshgrid = np.meshgrid(grid1, grid2, indexing='ij')
    values = fun(meshgrid)

    spline_fun = interpolate(values, meshgrid)

    x = np.random.rand(2, 5)
    assert np.allclose(spline_fun(x), fun(x), atol=1e-2, rtol=1e-2)

    x = np.random.rand(2, 5, 5)
    assert np.allclose(spline_fun(x), fun(x), atol=1e-2, rtol=1e-2)

    x = np.random.rand(2, 5, 5, 5)
    assert np.allclose(spline_fun(x), fun(x), atol=1e-2, rtol=1e-2)

    def fun(x):
        return (7 * x[0] * x[2] - np.power(x[1], 2)) * x[2]


    grid1 = np.arange(0, 1.0001, 0.01)
    grid2 = np.arange(0, 1.0001, 0.01)
    grid3 = np.arange(0, 1.0001, 0.01)

    meshgrid = np.meshgrid(grid1, grid2, grid3, indexing='ij')
    values = fun(meshgrid)

    spline_fun = interpolate(values, meshgrid)

    x = np.random.rand(3, 5)
    assert np.allclose(spline_fun(x), fun(x), atol=1e-2, rtol=1e-2)

    x = np.random.rand(3, 5, 5)
    assert np.allclose(spline_fun(x), fun(x), atol=1e-2, rtol=1e-2)

    x = np.random.rand(3, 5, 5, 5)
    assert np.allclose(spline_fun(x), fun(x), atol=1e-2, rtol=1e-2)
