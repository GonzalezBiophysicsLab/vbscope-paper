import numpy as np
import numba as nb

@nb.jit(nb.double[:,:](nb.double[:,:],nb.double[:,:,:]),nopython=True)
def update_tmatrix(alpha0,xi):
	n,k,_ = xi.shape
	alphan = np.zeros((k,k))
	for i in range(k):
		for j in range(k):
			alphan[i,j] = alpha0[i,j]
			for t in range(n):
				alphan[i,j] += xi[t,i,j]
	return alphan

@nb.jit(nb.double[:](nb.double[:],nb.double[:,:]),nopython=True)
def update_rho(rho0,gamma):
	rhon = np.zeros_like(rho0)
	for i in range(rho0.size):
		rhon[i] = rho0[i] + gamma[0,i]
	return rhon


@nb.jit(nb.types.Tuple((nb.double[:],nb.double[:],nb.double[:],nb.double[:]))(nb.double[:],nb.double[:,:],nb.double[:],nb.double[:],nb.double[:],nb.double[:]),nopython=True)
def update_normals(x,r,m0,beta0,a0,b0):
	n,k = r.shape

	mn = np.zeros_like(m0)
	betan = np.zeros_like(beta0)
	an = np.zeros_like(a0)
	bn = np.zeros_like(b0)


	nk = np.zeros(k) + 1e-300
	xk = np.zeros(k)
	sk = np.zeros(k)
	for i in range(k):
		for nn in range(n):
			nk[i] += r[nn,i]
			xk[i] += r[nn,i]*x[nn]
		xk[i] /= nk[i]
		for nn in range(n):
			sk[i] += r[nn,i]*(x[nn]-xk[i])**2.
		sk[i] /= nk[i]

	betan = beta0 + nk
	an = a0 + 0.5*nk
	mn = (beta0*m0+nk*xk)/betan
	bn = b0 + .5*nk*sk + .5*(beta0*nk)/betan * (xk - m0)**2.

	return mn,betan,an,bn
