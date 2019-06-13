import numpy as np
import numba as nb

@nb.njit(nb.double[:](nb.double[:],nb.double))
def gaussian_filter(x,sigma):
	g = np.zeros(x.size)
	l = int(sigma*3.)+1
	for i in range(x.size):
		ksum = 0.
		for j in range(i-l,i+l+1):
			if j >= 0 and j < x.size:
				kernel = np.exp(-.5/(sigma**2.)*(j-i)**2.)
				ksum += kernel
				g[i] += x[j]*kernel
		g[i] /= ksum
	return g

@nb.njit(nb.double[:](nb.double[:],nb.int64))
def minimum_filter(x,l):
	## minimium filter
	if l <= 0:
		return x
	m = np.zeros(x.size) + np.inf
	for i in range(x.size):
		for j in range(i-l,i+l+1):
			if j >= 0 and j < x.size:
				if x[j] < m[i]:
					m[i] = x[j]
	return m
