## Tools for numerical math calculations

[![Build Status](https://travis-ci.com/Bellator95/scikit-numerical.svg?branch=master)](https://travis-ci.com/Bellator95/scikit-numerical)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/e599de8c6de048ef8351811388c63632)](https://www.codacy.com/app/maksym.shpakovych/numerical?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Bellator95/scikit-numerical&amp;utm_campaign=Badge_Grade)
[![Codacy Badge](https://api.codacy.com/project/badge/Coverage/e599de8c6de048ef8351811388c63632)](https://www.codacy.com/app/maksym.shpakovych/numerical?utm_source=github.com&utm_medium=referral&utm_content=Bellator95/scikit-numerical&utm_campaign=Badge_Coverage)

This repository contains tools for math numerical computation such as numerical integration and interpolation. The current implementation contains:
-   numerical integration using Gauss formula

    ```python
    import numpy as np
    from numerical.integration import gauss
    
    def f(x):
        return np.power(x[0], 2)
    
    gauss.integrate(f, 0., 1.)  # 0.3333333
  
    ```


-   spline functions and theirs derivatives

    ```python
    ```python
        import numpy as np
        from numerical import spline
        from numerical import diff
        import matplotlib.pyplot as plt
            
        x = np.arange(0, 4., 0.05)
        y = spline.schoenberg1d(x)
        d1spline = diff(spline.schoenberg1d, "x", 1) # first derivative
        yd1 = d1spline(x)
        d2spline = diff(spline.schoenberg, "x", 2) # second derivative
        yd2 = d2spline(x)  
        # visualize results
        plt.plot(x, y)
        plt.plot(x, yd1)
        plt.plot(x, yd2)
        plt.show()
          
        ```
    ```

![spline_derivs](https://github.com/Bellator95/scikit-numerical/blob/master/images/shenberg_spline_derivatives.png)


-   function interpolation

    ```python
    import numpy as np
    from numerical import interpolate
    import matplotlib.pyplot as plt
    
    def fun(x):
        return 1 - np.power(x[0] - 0.5, 2)

    grid = np.array([np.arange(0, 1.0001, 0.25)])
    values = fun(grid)
    itp_fun = interpolate(values, grid)
    
    x = np.arange(0., 1.00001, 0.001).reshape(1, -1)
    y_intp = itp_fun(x)
    y_true = fun(x)

    plt.plot(x[0], y_intp)
    plt.plot(x[0], y_true)
    plt.show()
    ```
    
![linear_interpolation](https://github.com/Bellator95/scikit-numerical/blob/master/images/linear_interpolation.png)
