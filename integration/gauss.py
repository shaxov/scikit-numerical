import numpy as np
from nles.utils.area.boundary import Boundary
from numpy.polynomial.legendre import leggauss

from nles.utils.area.grid import Grid


def gl_integrate(func, grid, n_roots, batch_size):
    """ Gauss-Legendre function integration

    Attributes:
        func: function which takes numpy.ndarray with first shape equal to n, where n - count of variables (max 3).
        grid: nles.utils.area.grid.Grid object which contains data for numerical integration.
        n_roots: count of zero roots in Legendre polynomial.
        batch_size: tuple, batch size for integration process

    Returns:
        numpy.ndarray values of function integral in grid area.
    """
    (roots, weights) = leggauss(n_roots)
    roots = roots.reshape(1, -1)

    result = []
    start = [0] * grid.dim
    w = _multi_dot(*([weights] * grid.dim)).ravel().reshape(-1, 1)
    outer_diff = _multi_dot(*[grid1d['diff'] for grid1d in grid]).ravel()

    _integration_loop(grid, func, w, roots, n_roots, start, batch_size, result)
    mm = np.concatenate(result)
    return np.dot(outer_diff, mm)[0]


def _multi_dot(*vectors):
    """ Pairwise vectors product.

    Args:
        vectors: tuple of numpy.array with len(shape) = 1
    Returns:
        numpy.ndarray
    """
    if len(vectors) == 1:
        return vectors[0]
    vec_dot = np.dot(np.expand_dims(vectors[0], -1), np.expand_dims(vectors[1], 0))
    return _multi_dot(vec_dot, *vectors[2:])


def _repeat(f_args, n_roots=None, cnt_reps=None):
    """ Repeat arguments as cartesian product.

    Args:
        f_args: list of arguments for repeating.
        n_roots: int, number of roots in gauss-legender quadrature.
        cnt_reps: list of int, repeat count for each argument.
    Returns:
        list of numpy.ndarray of repeated arguments' value.
    """
    if len(f_args) == 1:
        return f_args

    elif len(f_args) == 2:
        xx_rep = np.repeat(np.repeat(f_args[0], n_roots, axis=1),
                           cnt_reps[1], axis=0)
        yy_rep = np.tile(np.tile(f_args[1], (1, n_roots)),
                         (cnt_reps[0], 1))
        return [xx_rep, yy_rep]

    elif len(f_args) == 3:
        xx_rep = np.repeat(np.repeat(f_args[0], n_roots * n_roots, axis=1),
                           cnt_reps[1] * cnt_reps[2], axis=0)
        yy_rep = np.repeat(np.tile(np.tile(f_args[1], (1, n_roots * n_roots)), (cnt_reps[0], 1)),
                           cnt_reps[2], axis=0)
        zz_rep = np.tile(np.tile(f_args[2], (1, n_roots * n_roots)),
                         (cnt_reps[0] * cnt_reps[1], 1))
        return [xx_rep, yy_rep, zz_rep]

    else:
        raise ValueError("Repeating arguments for dims > 3 is not implemented.")


def _integration_loop(grid, func, w, roots, n_roots, _start, batch_size, result_array: list, nest_id=-1):
    if nest_id == grid.dim - 1:
        args = [np.add(grid[i]['sum'][_start[i]:_start[i] + batch_size[i]],
                       np.dot(grid[i]['diff_t'][_start[i]:_start[i] + batch_size[i]], roots))
                for i in range(grid.dim)]

        cnt_reps = [len(grid[i]['sum'][_start[i]:_start[i] + batch_size[i]]) for i in range(grid.dim)]
        f_val = func(np.array(_repeat(args, n_roots, cnt_reps)))

        result_array.append(np.matmul(f_val, w))
    else:
        nest_id += 1
        while _start[nest_id] < grid[nest_id]['n_roots']:
            _integration_loop(grid, func, w, roots, n_roots, _start, batch_size, result_array, nest_id)
            _start[nest_id] += batch_size[nest_id]
        _start[nest_id] = 0


if __name__ == '__main__':
    from time import perf_counter


    def f(x):
        return np.power(x[0], 2) + 2 * x[0] + 5


    def f2(x):
        return np.power(x[0], 2) + 2 * x[1] + 5


    grid = Grid([Boundary(-3.24, 9.24)], [0.02])
    print(grid)
    start = perf_counter()
    assert np.allclose(gl_integrate(f, grid, 16, (64,)), [411.580])
    print(perf_counter() - start, '\n')

    boundary_x = Boundary(0, 1)
    boundary_y = Boundary(0, 2 * np.pi)
    grid = Grid([boundary_x, boundary_y], [0.02, np.pi / 8])

    print(grid)
    start = perf_counter()
    assert np.allclose(gl_integrate(f2, grid, 16, (64, 64)), [72.9887])
    print(perf_counter() - start, '\n')

    boundary_x = Boundary(-1, 1.5)
    boundary_y = Boundary(-0.2, 0.5)
    boundary_z = Boundary(0, 1.8)
    grid = Grid([boundary_x, boundary_y, boundary_z], [0.05, 0.05, 0.05])


    def f(x):
        return (7 * x[0] * x[2] - np.power(x[1], 2)) * x[2]


    print(grid)
    start = perf_counter()
    val = gl_integrate(f, grid, 16, (128, 128, 128))
    print(f"Integral value: {val}")
    assert np.allclose(val, [5.77375])
    print(f"Calculation time: {perf_counter() - start:.2f}s")
