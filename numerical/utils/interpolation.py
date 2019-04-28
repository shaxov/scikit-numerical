import numpy as np


def repeat_args(args, nodes_count):
    """ Repeat arguments as cartesian product.

        Args:
            args: list of arguments for repeating.
            nodes_count: tuple, nodes of grid
        Returns:
            numpy.ndarray of repeated arguments' value.
        """
    if len(args) == 1:
        return args
    elif len(args) == 2:
        d1 = np.repeat(args[0], nodes_count[1], -1)
        tile_dims = [1] * len(args[1].shape)
        tile_dims[-1] = nodes_count[0]
        d2 = np.tile(args[1], tuple(tile_dims))
        dd1 = np.expand_dims(d1, -1)
        dd2 = np.expand_dims(d2, -1)
        return np.rollaxis(np.concatenate([dd1, dd2], -1), -1, 0)
    elif len(args) == 3:
        d1 = np.repeat(args[0], (nodes_count[1]) * (nodes_count[2]), axis=-1)
        tile_dims = [1] * len(args[1].shape)
        tile_dims[-1] = nodes_count[0]
        d2 = np.repeat(np.tile(args[1], tuple(tile_dims)), nodes_count[2], axis=-1)
        tile_dims = [1] * len(d2.shape)
        tile_dims[-1] = (nodes_count[0]) * (nodes_count[1])
        d3 = np.tile(args[2], tuple(tile_dims))
        dd1 = np.expand_dims(d1, -1)
        dd2 = np.expand_dims(d2, -1)
        dd3 = np.expand_dims(d3, -1)
        return np.rollaxis(np.concatenate([dd1, dd2, dd3], -1), -1, 0)
    else:
        raise ValueError("Repeat arguments are not implemented for dim > 3")
