from numerical.splines.derivatives import schoenberg_spline_derivatives


class setup:
    def __init__(self, ftype: str, max_order: [int, tuple] = None):
        self.ftype = ftype
        self.max_order = max_order

    def __call__(self, f):
        if self.ftype == "numerical":
            if f.__name__ == "schoenberg":
                f = schoenberg_spline_derivatives(f)
            else:
                raise NotImplementedError
        elif self.ftype == "symbolical":
            raise NotImplementedError
        else:
            raise ValueError(f"ftype '{self.ftype}' is not valid. Please use 'numerical' or 'symbolical' type.")
        return f
