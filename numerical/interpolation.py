import numpy as np

from numerical import splines
from numerical.utils.interpolation import repeat_args


def interpolate(values: np.array, nodes: list, batch_size=16, bfunc=None):
    """ Builds function which is an interpolation function on nodes with computer values in these nodes.

    Args:
        values: list of function values in grid nodes.
        nodes: points where function was calculated.
        batch_size: int, batch size for interpolation process.
        bfunc: basis function which is used to interpolate spaces between nodes.

    Returns:
        interpolated function.
    """
    # if basis function is not set, linear interpolation will be used
    if bfunc is None:
        bfunc = splines.linear

    nodes_range = [(dim.min(), dim.max()) for dim in nodes]
    nodes_count = nodes[0].shape
    nodes_dim = len(nodes)

    shift_indexes = [np.arange(0, dim, dtype=np.float64) for dim in nodes_count]

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
        return result

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
        spline_val = np.prod([bfunc(arg) for arg in rep_args], axis=0)
        result.append(np.dot(spline_val, values))
        batch_position += batch_size


if __name__ == '__main__':
    def fun(x):
        return np.power(x, 2) + 4

    nodes = [np.arange(0, 9.20001, 0.2)]
    values = fun(nodes[0])

    spline_fun = interpolate(values, nodes)
    assert np.allclose(spline_fun(np.array([[0.4684]])), np.float64(4.2284), atol=1e-8)
    assert np.allclose(spline_fun(np.array([[2.6594]])), np.float64(11.0808), atol=1e-8)

    # boundary_x = Boundary(0, 1)
    # boundary_y = Boundary(0, 1)
    # boundary_z = Boundary(0, 1)
    # gridd = Grid([boundary_x, boundary_y, boundary_z], [0.01, 0.01, 0.01])
    #
    #
    def fun(x):
        return (7 * x[0] * x[2] - np.power(x[1], 2)) * x[2]
    #
    # #tabs = func(grid['mesh'].T)
    # spline_fun = bspline_interpolation(tabs, grid, 16)
    #
    # assert np.allclose(spline_fun(np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])), [0.75])
    # assert np.allclose(spline_fun(np.array([[[0.25]], [[0.786]], [[0.3335]]])), [-0.01136422])