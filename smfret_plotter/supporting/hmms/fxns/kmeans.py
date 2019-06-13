import numpy as np
import numba as nb

@nb.njit(nb.types.Tuple((nb.int64[:,:],nb.double[:],nb.double[:],nb.double[:]))(nb.double[:],nb.int64))
def kmeans(x,nstates):
	nx = x.size

	xmin = x.min()
	xmax = x.max()
	mu = np.linspace(xmin,xmax,nstates)
	# mu = np.random.rand(nstates)*(xmax-xmin) + xmin
	# mu = x[np.random.randint(0,x.shape[0],size=nstates)]

	ll_last = np.inf
	dist = np.zeros((x.size,mu.size),dtype=nb.double)
	r = np.zeros((x.size,mu.size),dtype=nb.int64)

	for iteration in range(500):
		for i in range(nx):
			dmin = 0
			for j in range(nstates):
				dist[i,j] = np.sqrt((x[i]-mu[j])**2.)
				if dist[i,j] < dist[i,dmin]:
					dmin = j
				r[i,j] = 0
			r[i,dmin] = 1

		for j in range(nstates):
			rr = r[:,j]
			mu[j] = float(np.sum(rr*x))/(np.sum(rr)+1e-16)

		ll = np.sum(r*dist)
		if np.abs((ll - ll_last)/ll) <= 1e-100:
			break
		else:
			ll_last = ll

	pi = np.zeros_like(mu)
	for i in range(nx):
		for j in range(nstates):
			pi[j] += r[i,j]

	# xkeep = pi > 0
	# r = r[:,xkeep]
	# mu = mu[xkeep]
	# pi = pi[xkeep]

	var = np.zeros_like(mu)
	nn = np.zeros_like(mu) + 1e-6
	mm = np.zeros_like(mu)
	for i in range(nx):
		for j in range(nstates):
			r2 = float(r[i,j]) + 1./float(r.shape[0])
			# print(r2)
			mm[j] += r2 * x[i]
			var[j] += r2 * x[i]**2.
			nn[j] += r2
			# if r[i,j] == 1:
				# var[j] += r[i,j]*x[i]**2.
				# nn[j] += 1
	mm /= nn
	var /= nn
	var -= mm**2.
	pi = nn/np.sum(nn)
	# var /= nn
	# var -= mu**2.
	# pi /= float(nx)
	# return r,mu,var,pi
	return r,mm,var,pi


if __name__ == '__main__':
	import matplotlib.pyplot as plt
	np.random.seed(666)

	d = np.concatenate([np.random.normal(size=np.random.randint(low=50,high=500))+np.random.randint(low=-100,high=100) for _ in range(3)])

	r,mu,var,pi = kmeans(d,4)
	print(mu)
	print(var)
	print(pi)

	t = np.arange(d.size)
	for i in range(mu.size):
		xkeep = r.argmax(1) == i
		plt.plot(t[xkeep],d[xkeep],ls='None',marker='o',alpha=.05)
	plt.show()
