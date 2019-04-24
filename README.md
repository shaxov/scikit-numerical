## Tools for numerical math calculations

[![Build Status](https://travis-ci.com/Bellator95/scikit-numerical.svg?branch=master)](https://travis-ci.com/Bellator95/scikit-numerical)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/e599de8c6de048ef8351811388c63632)](https://www.codacy.com/app/maksym.shpakovych/numerical?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Bellator95/scikit-numerical&amp;utm_campaign=Badge_Grade)
[![Codacy Badge](https://api.codacy.com/project/badge/Coverage/e599de8c6de048ef8351811388c63632)](https://www.codacy.com/app/maksym.shpakovych/numerical?utm_source=github.com&utm_medium=referral&utm_content=Bellator95/scikit-numerical&utm_campaign=Badge_Coverage)

This repository contains tools for math numerical computation such as numerical integration and interpolation. The current implementation contains:

- numerical integration using Gauss formula

```python
import numpy as np
from numerical.integration import gauss
from numerical.area import bounds
from numerical.area import grid

def f(x):
    return np.power(x[0], 2)

bd = bounds.cartesian.LineBoundary1D(0, 1)
gd = grid.UniformGrid(bd, step=[0.02])
gauss.integrate(f, gd)
```