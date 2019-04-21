import numpy as np
from numpy.polynomial.legendre import leggauss

from area.grid import Grid
from utils.linalg import multi_dot2


def integrate(ndfunc: "numpy function", ndgrid: Grid, roots_count: int = 32, batch_size: tuple = None):
    """ Integrate a function numerically using Gauss formula

    Args:
        ndfunc: function which takes numpy.ndarray where the first shape equal to n, where n is a count of variables (max 3).
        ndgrid: area.grid.Grid object which contains grid data for numerical integration.
        roots_count: count of zero roots in Legendre polynomial.
        batch_size: tuple, batch size for integration process

    Returns:
        numpy.ndarray values of function integral in grid area.
    """
    if batch_size is None:
        batch_size = (32,) * ndgrid.dim

    leg_roots, leg_weights = leggauss(roots_count)
    leg_roots = leg_roots.reshape(1, -1)

    result = []
    # positions of batch step for each dimension
    batch_position = [0] * ndgrid.dim
    # pairwise product of legendre weights for function evaluation
    nd_leg_weights = multi_dot2(*(ndgrid.dim * [leg_weights]), flatten=True, reshape=True)
    # pairwise product of grid steps diff for the left function multiplication
    nd_outer_diff = multi_dot2(*[gd['diff'] for gd in ndgrid], flatten=True)
    # batch integration recursive loop with output in 'results'
    _integration_loop(result, ndfunc, ndgrid, nd_leg_weights, leg_roots, roots_count, batch_position, batch_size)
    result = np.concatenate(result)
    return np.dot(nd_outer_diff, result)[0]


def _repeat(f_args, n_roots, cnt_reps):
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
        xx_rep = np.repeat(np.repeat(f_args[0], n_roots, axis=1), repeats=cnt_reps[1], axis=0)
        yy_rep = np.tile(f_args[1], (cnt_reps[0], n_roots))
        return np.array([xx_rep, yy_rep])

    elif len(f_args) == 3:
        xx_rep = np.repeat(np.repeat(f_args[0], n_roots * n_roots, axis=1), repeats=cnt_reps[1] * cnt_reps[2], axis=0)
        yy_rep = np.repeat(np.tile(f_args[1], (cnt_reps[0], n_roots * n_roots)), repeats=cnt_reps[2], axis=0)
        zz_rep = np.tile(f_args[2], (cnt_reps[0] * cnt_reps[1], n_roots * n_roots))
        return np.array([xx_rep, yy_rep, zz_rep])

    else:
        raise ValueError("Repeating arguments for dims > 3 is not implemented.")


def _integration_loop(result_list, ndfunc, ndgrid, nd_leg_weights, leg_roots, roots_count,
                      batch_position, batch_size, nest_id=-1):
    if nest_id == ndgrid.dim - 1:
        batch_args = []
        cnt_reps = []
        for dim in range(ndgrid.dim):
            dim_batch_position = batch_position[dim]
            sum_batch = ndgrid[dim]['sum'][dim_batch_position:dim_batch_position + batch_size[dim]]
            diff_t_batch = ndgrid[dim]['diff_t'][dim_batch_position:dim_batch_position + batch_size[dim]]
            # i-dim batch function argument
            batch_arg = sum_batch + diff_t_batch @ leg_roots

            batch_args.append(batch_arg)
            cnt_reps.append(len(sum_batch))

        f_val = ndfunc(_repeat(batch_args, roots_count, cnt_reps))
        fw_mul = np.matmul(f_val, nd_leg_weights)
        result_list.append(fw_mul)
    else:
        nest_id += 1
        while batch_position[nest_id] < ndgrid[nest_id]['n_roots']:
            _integration_loop(result_list, ndfunc, ndgrid, nd_leg_weights, leg_roots, roots_count,
                              batch_position, batch_size, nest_id)
            batch_position[nest_id] += batch_size[nest_id]
        batch_position[nest_id] = 0


if __name__ == '__main__':
    from time import perf_counter
    from area.grid import Grid
    from area.boundary import Boundary


    def f(x):
        return np.power(x[0], 2) + 2 * x[0] + 5


    def f2(x):
        return np.power(x[0], 2) + 2 * x[1] + 5


    grid = Grid([Boundary(-3.24, 9.24)], [0.02])
    print(grid)
    start = perf_counter()
    assert np.allclose(integrate(f, grid, 16, (64,)), [411.580])
    print(perf_counter() - start, '\n')

    boundary_x = Boundary(0, 1)
    boundary_y = Boundary(0, 2 * np.pi)
    grid = Grid([boundary_x, boundary_y], [0.02, np.pi / 8])

    print(grid)
    start = perf_counter()
    assert np.allclose(integrate(f2, grid, 16, (64, 64)), [72.9887])
    print(perf_counter() - start, '\n')

    boundary_x = Boundary(-1, 1.5)
    boundary_y = Boundary(-0.2, 0.5)
    boundary_z = Boundary(0, 1.8)
    grid = Grid([boundary_x, boundary_y, boundary_z], [0.05, 0.05, 0.05])


    def f(x):
        return (7 * x[0] * x[2] - np.power(x[1], 2)) * x[2]


    print(grid)
    start = perf_counter()
    val = integrate(f, grid, 16, (32, 32, 32))
    print(f"Integral value: {val}")
    assert np.allclose(val, [5.77375])
    print(f"Calculation time: {perf_counter() - start:.2f}s")
