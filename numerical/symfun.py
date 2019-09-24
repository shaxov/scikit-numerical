import numexpr as ne


class SymbolicFunction():
    def __init__(self, expr):
        self.expr = expr
        
    def __call__(self, **kwargs):
        return ne.evaluate(self.expr, local_dict=kwargs)
    
    def __repr__(self):
        return self.expr


if __name__ == '__main__':
    sf = SymbolicFunction("x**2 + x*y*5 - z + 15")
    print(sf)
    print(sf(x=[5], y=[7], z=[12]))
