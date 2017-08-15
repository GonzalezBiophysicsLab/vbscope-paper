#viterbi numba
import numba as nb
import numpy as np

@nb.jit(nb.int64[:](nb.double[:,:],nb.double[:,:],nb.double[:]),nopython=True)
def viterbi(p_x_z,A,pi):
	n,k = p_x_z.shape

	omega = np.zeros((n,k))
	zmax = np.zeros((n,k),dtype=nb.int64)
	zhat = np.zeros(n,dtype=nb.int64)

	for i in range(k):
		omega[0,i] = pi[i] + p_x_z[0,i]

	for t in range(1,omega.shape[0]):
		for i in range(k):
			q = A[0,i] + omega[t-1][0]
			for j in range(k):
				if A[j,i] + omega[t-1][j] > q:
					q = A[j,i] + omega[t-1][j]
			q = np.max(A[:,i]+omega[t-1])
			omega[t,i] = q + p_x_z[t,i]

			for j in range(k):
				if A[j,i] + omega[t-1,j] > A[zmax[t,i],i] + omega[t-1,zmax[t,i]]:
					zmax[t,i] = j

	for i in range(k):
		if omega[n-1,zhat[n-1]] < omega[n-1,i]:
			zhat[n-1] = i
	for tt in range(n-1):
		t = n-2-tt
		zhat[t] = zmax[t+1,zhat[t+1]]

	return zhat
