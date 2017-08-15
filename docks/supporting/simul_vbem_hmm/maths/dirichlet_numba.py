import numpy as np
import numba as nb
from math_fxns import psi

@nb.jit(nb.double[:](nb.double[:]),nopython=True)
def dirichlet_estep(x):
	s = np.sum(x)
	return psi(x) - psi(s)
