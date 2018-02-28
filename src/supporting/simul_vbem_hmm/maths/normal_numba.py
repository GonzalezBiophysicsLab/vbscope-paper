import numpy as np
import numba as nb

@nb.jit(nb.double[:,:](nb.double[:],nb.double[:],nb.double[:]),nopython=True)
def ln_p_normal(x,mu,var):
	out = np.zeros((x.size,mu.size))
	for j in range(mu.size):
		if var[j] > 0:
			for i in range(x.size):
				out[i,j] = -.5*np.log(2.*np.pi) -.5*np.log(var[j]) - .5/var[j]*(x[i]-mu[j])**2.
		else:
			for i in range(x.size):
				out[i,j] = -np.inf
	return out

@nb.jit(nb.double[:,:](nb.double[:],nb.double[:],nb.double[:]),nopython=True)
def p_normal(x,mu,var):
	return np.exp(ln_p_normal(x,mu,var))
