import numpy as np
from numerical.utils.linalg import polar2cartesian


def repeat_args(f_args, n_roots, cnt_reps):
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


def coordinate_transform(ndfunc, coords_type):
    if coords_type == "cartesian":
        return ndfunc
    elif coords_type == "polar":
        def ndfunc_t(x):
            return ndfunc(polar2cartesian(x)) * x[0]
        return ndfunc_t
    elif coords_type == "spherical":
        raise NotImplementedError("Spherical coordinates was not implemented.")
    else:
        raise ValueError("Coordinates type can be 'cartesian', 'polar' or 'spherical'.")
