import numpy as np


class Boundary(object):
    """ Class represents start and end of 1-dim line. """
    def __init__(self, start, end):
        self.start = np.float64(start)
        self.end = np.float64(end)

    def __repr__(self):
        return f"<Boundary: start={self.start} end={self.end}>"
