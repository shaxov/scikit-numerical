import sympy
from numerical import symfun
from numerical import spline_derivs


class setup:
    def __init__(self, var_names):
        self._var_names = var_names

    def __call__(self, func):
        if func.__name__ == "schoenberg1d":
            func = spline_derivs.schoenberg1d(func, self._var_names)
        elif func.__name__ == "schoenberg2d":
            func = spline_derivs.schoenberg2d(func, self._var_names)
        else:
            raise NotImplementedError("Derivatives for function '{}' are not implemented".format(func.__name__))
        return func


def diff(func, *args):
    if isinstance(func, symfun.SymbolicFunction):
        func = symfun.SymbolicFunction(str(sympy.diff(func.expr, *args)))
    else:
        func = func.deriv(*args)
    return func
