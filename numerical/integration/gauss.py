import numpy as np
from numpy.polynomial.legendre import leggauss
from numerical.utils.linalg import multi_dot2
from numerical.utils.integration import repeat_args, coordinate_transform
from numerical.area import grid


def integrate(ndfunc: "numpy function",
              *bounds: "integration limits",
              steps: tuple = (),
              coords_type="cartesian",
              ndgrid: "area grid" = None,
              roots_count: int = 32,
              batch_size: tuple = None):
    """ Integrate a function numerically using Gauss formula

    Args:
        ndfunc: function which takes numpy.ndarray where the first shape equal to n, where n is a count of variables 1<= n <=3.
        bounds: tuple of floats which indicate integration limits
        steps: tuple of floats which indicates integration steps
        coords_type: str, type of coordinates for integration ('cartesian', 'polar', 'spherical')
        ndgrid: area.grid.Grid object which contains grid data for numerical integration.
        roots_count: count of zero roots in Legendre polynomial.
        batch_size: tuple, batch size for integration process

    Returns:
        numpy.ndarray values of function integral in grid area.
    """

    if bounds:
        ndgrid = grid.UniformGrid(bounds, steps)

    if batch_size is None:
        batch_size = (32,) * ndgrid.dim

    ndfunc = coordinate_transform(ndfunc, coords_type)
    _build_integration_meta(ndgrid)

    leg_roots, leg_weights = leggauss(roots_count)
    leg_roots = leg_roots.reshape(1, -1)

    result = []
    # positions of batch step for each dimension
    batch_position = [0] * ndgrid.dim
    # pairwise product of legendre weights for function evaluation
    nd_leg_weights = multi_dot2(*(ndgrid.dim * [leg_weights]), flatten=True, reshape=True)
    # pairwise product of grid steps diff for the left function multiplication
    nd_outer_diff = multi_dot2(*[gd.diff for gd in ndgrid], flatten=True)
    # batch integration recursive loop with output in 'results'
    _integration_loop(result, ndfunc, ndgrid, nd_leg_weights, leg_roots, roots_count, batch_position, batch_size)
    result = np.concatenate(result)
    return np.dot(nd_outer_diff, result)[0]


def _integration_loop(result_list, ndfunc, ndgrid, nd_leg_weights, leg_roots, roots_count,
                      batch_position, batch_size, nest_id=-1):
    if nest_id == ndgrid.dim - 1:
        batch_args = []
        cnt_reps = []
        for dim in range(ndgrid.dim):
            dim_batch_position = batch_position[dim]
            sum_batch = ndgrid[dim].sum[dim_batch_position:dim_batch_position + batch_size[dim]]
            diff_t_batch = ndgrid[dim].diff_t[dim_batch_position:dim_batch_position + batch_size[dim]]
            # i-dim batch function argument
            batch_arg = sum_batch + diff_t_batch @ leg_roots

            batch_args.append(batch_arg)
            cnt_reps.append(len(sum_batch))

        f_val = ndfunc(repeat_args(batch_args, roots_count, cnt_reps))
        fw_mul = np.matmul(f_val, nd_leg_weights)
        result_list.append(fw_mul)
    else:
        nest_id += 1
        while batch_position[nest_id] < ndgrid[nest_id].nodes_count:
            _integration_loop(result_list, ndfunc, ndgrid, nd_leg_weights, leg_roots, roots_count,
                              batch_position, batch_size, nest_id)
            batch_position[nest_id] += batch_size[nest_id]
        batch_position[nest_id] = 0


def _build_integration_meta(ndgrid):
    for g in ndgrid:
        g.sum = ((g.nodes[1:] + g.nodes[:-1]) / 2.0).reshape(-1, 1)
        g.diff = (g.nodes[1:] - g.nodes[:-1]) / 2.0
        g.diff_t = g.diff.reshape(-1, 1)
        g.nodes_count = len(g.nodes)
