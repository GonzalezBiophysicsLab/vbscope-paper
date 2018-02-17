import numpy as np
import numba as nb
from math_fxns import psi
from dirichlet_numba import dirichlet_estep

@nb.jit(nb.double[:,:](nb.double[:],nb.double[:],nb.double[:],nb.double[:],nb.double[:]),nopython=True)
def vbem_1d_ln_p_x_z(x,mk,betak,ak,bk):

	n = x.size
	k = mk.size

	ln_p_x_z = np.zeros((n,k),dtype=nb.double)

	pre = -.5*np.log(2.*np.pi)
	for i in range(k):
		ln_lam = psi(ak[i]) - np.log(bk[i])
		for nn in range(n):
			ln_p_x_z[nn,i] = pre + ln_lam  - (1./betak[i] + ak[i]/bk[i]*(x[nn]-mk[i])**2.)

	return ln_p_x_z


@nb.jit(nb.double[:,:](nb.double[:,:]),nopython=True)
def vbem_1d_ln_A(alphakl):

	k,_ = alphakl.shape
	ln_A = np.zeros((k,k),dtype=nb.double)

	for i in range(k):
		ln_A[i] = dirichlet_estep(alphakl[i])

	return ln_A

@nb.jit(nb.double[:](nb.double[:]),nopython=True)
def vbem_1d_ln_pi(rhok):
	return dirichlet_estep(rhok)
