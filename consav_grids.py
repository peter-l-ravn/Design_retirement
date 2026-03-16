import numpy as np
from numba import njit
from jit_module import jit_if_enabled

@jit_if_enabled(fastmath=False)
def nonlinspace(x_min,x_max,n,phi):
    """ like np.linspace. but with unequal spacing

    Args:

        x_min (double): minimum value
        x_max (double): maximum value
        n (int): number of points
        phi (double): phi = 1 -> eqaul spacing, phi up -> more points closer to minimum
    
    Returns:

        y (list): grid with unequal spacing

    """

    assert x_max > x_min
    assert n >= 2
    assert phi >= 1
 
    # 1. recursion
    y = np.empty(n)
 
    y[0] = x_min
    for i in range(1, n):
        y[i] = y[i-1] + (x_max-y[i-1]) / (n-i)**phi
    
    # 3. assert increaing
    assert np.all(np.diff(y) > 0)
 
    return y