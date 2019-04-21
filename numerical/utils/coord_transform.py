""" Module provides coordinates transformation operations.

Available transformations:
    - cartesian coordinates to polar.
    - polar coordinates to cartesian.
"""

import numpy as np


def _argument(x):
    """ Return argument of complex number (phi angle) for vectors. """
    fi = np.zeros(shape=x.shape[1:], dtype=np.float64)
    pos_idx = np.where(x[1] >= 0)
    neg_idx = np.where(x[1] < 0)
    fi[pos_idx] = np.angle(x[0][pos_idx] + 1j * x[1][pos_idx])
    fi[neg_idx] = 2*np.pi + np.angle(x[0][neg_idx] + 1j * x[1][neg_idx])
    return fi


def _absolute_value(x):
    """ Return absolute value for vectors.

    Euclidean norm.
    """
    return np.linalg.norm(x, axis=0)


def polar2cartesian(x):
    """ Transform polar coordinates to cartesian.

    Coordinates order:
        x[0] ~ ro
        x[1] ~ phi
    """
    return np.array([x[0] * np.cos(x[1]), x[0] * np.sin(x[1])], dtype=np.float64)


def cartesian2polar(x):
    """ Transform cartesian coordinates to polar. """
    return np.array([_absolute_value(x), _argument(x)], dtype=np.float64)
