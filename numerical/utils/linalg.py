import numpy as np


def multi_dot(*vectors):
    """ Pairwise vectors product.

    Args:
        vectors: tuple of numpy.array with len(shape) = 1
    Returns:
        numpy.ndarray
    """
    if len(vectors) == 1:
        return vectors[0]
    vec_dot = np.dot(np.expand_dims(vectors[0], -1), np.expand_dims(vectors[1], 0))
    return multi_dot(vec_dot, *vectors[2:])


def multi_dot2(*vectors, flatten=False, reshape=False):
    md = multi_dot(*vectors)
    if flatten:
        md = md.ravel()
    if reshape:
        md = md.reshape(-1, 1)
    return md
