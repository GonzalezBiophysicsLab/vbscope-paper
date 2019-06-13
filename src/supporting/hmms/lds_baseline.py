import numpy as np
import numba as nb

@nb.njit(nb.double[:,:](nb.int64,nb.double,nb.double))
def fake_data(n,sigma_n,sigma_b):
	np.random.seed(666)
	d = np.random.normal(loc=0.,scale=sigma_b,size=n).cumsum()
	b = np.random.normal(loc=0.,scale=sigma_n,size=n)
	out = np.zeros((2,d.size))
	out[0] = d
	out[1] = d+b
	return out

@nb.njit(nb.double[:](nb.double[:]))
def initial_guess(x):

	xx = np.zeros_like(x)
	for i in range(x.size):
		xmin = i-5
		if xmin < 0:
			xmin = 0
		xmax = i + 5
		if xmax > x.size:
			xmax = x.size
		xx[i] = np.mean(x[xmin:xmax])

	mu0 = x[0]
	P0 = np.var(xx[:10])#np.sqrt(np.abs(xx[0]-xx[1]))

	var_d = np.var(xx)
	var_delta = np.var(xx[1:]-xx[:-1])

	N = xx.size
	Sigma = (var_d - N*var_delta)/(1.-2.*N)
	Gamma = var_delta - 2.*Sigma
	if Gamma < 0:
		Gamma = var_delta/10.

	return np.array((mu0,P0,Gamma,Sigma))

@nb.njit(nb.types.Tuple((nb.double[:],nb.double[:],nb.double[:,],nb.int64,nb.double[:]))(nb.double[:]))
def LDS(x):
	## Following Bishop 13.3
	## Fixing A = 1, C = 1

	Theta = initial_guess(x)
	mu0,P0,Gamma,Sigma = Theta

	maxiters = 1000
	threshold = 1e-5
	R2 = np.zeros(maxiters)

	mu = np.zeros_like(x)
	mu_hat = np.zeros_like(x)
	V = np.zeros_like(x)
	V_hat = np.zeros_like(x)
	K = np.zeros_like(x)
	P = np.zeros_like(x)
	J = np.zeros_like(x)
	E_z = np.zeros_like(x)
	E_zy = np.zeros_like(x)
	E_yz = np.zeros_like(x)
	E_zz = np.zeros_like(x)

	iteration = 0
	while iteration < maxiters:

		## Forward Pass
		## Initialize Recursion
		## 13.94-97
		K[0] = P0/(P0 + Sigma)
		mu[0] = mu0 + K[0]*(x[0] - mu0)
		V[0] = (1. - K[0])*P0
		P[0] = V[0] + Gamma

		## 13.89-92
		for i in range(1,x.size):
			K[i] = P[i-1]/(P[i-1] + Sigma)
			mu[i] = mu[i-1] + K[i]*(x[i] - mu[i-1])
			V[i] = (1. - K[i])*P[i-1]
			P[i] = V[i] + Gamma

		## Backward Pass
		## 13.100-102
		#### beta[-1] = 1... therefore mu_hat[-1] = mu[-1], V_hat[-1] = V[-1] (c.f. 13.98)
		mu_hat[-1] = mu[-1]
		V_hat[-1] = V[-1]
		J[-1] = V[-1]/P[-1]

		for j in range(x.size-1):
			i = x.size-2-j
			J[i] = V[i]/P[i]
			mu_hat[i] = mu[i] + J[i]*(mu_hat[i+1]-mu[i])
			V_hat[i] = V[i] + J[i]*(V_hat[i+1]-P[i])*J[i]

		## Expectations
		## 13.104-107
		E_z[0] = mu_hat[0]
		E_zz[0] = V_hat[0] + mu_hat[0]**2.
		for i in range(1,x.size):
			E_z[i] = mu_hat[i]
			E_zy[i] = V_hat[i]*J[i-1] + mu_hat[i]*mu_hat[i-1]
			E_yz[i] = J[i-1]*V_hat[i] + mu_hat[i-1]*mu_hat[i]
			E_zz[i] = V_hat[i] + mu_hat[i]**2.

		## Maximizations
		## 13.110-111, 13.114, 13.116
		mu0 = E_z[0]
		P0 = E_zz[0] - E_z[0]**2.
		Gamma = 0.
		Sigma = x[0]**2. - E_z[0]*x[0] - x[0]*E_z[0] + E_zz[0]
		for i in range(1,x.size):
			Gamma += E_zz[i] - E_yz[i] - E_zy[i] + E_zz[i-1]
			Sigma += x[i]**2. - E_z[i]*x[i] - x[i]*E_z[i] + E_zz[i]
		Gamma /= float(x.size) - 1.
		Sigma /= float(x.size)

		R2[iteration] = Gamma + Sigma
		if iteration > 1:
			if np.abs(R2[iteration] - R2[iteration-1])/R2[iteration] < threshold:
				break
		iteration += 1

	Theta_new = np.array((mu0,P0,Gamma,Sigma))

	return Theta_new,E_z,V_hat,iteration,R2[:iteration+1]

@nb.njit(nb.double[:](nb.double[:],nb.int64,nb.int64))
def filters(x,n,m):
	## minimium filter
	l = x.size//n
	if l <= 0:
		l = 5
	b = np.zeros(x.size) + np.inf
	for i in range(x.size):
		for j in range(i-l,i+l+1):
			if j >= 0 and j < x.size:
				if x[j] < b[i]:
					b[i] = x[j]

	## uniform filter
	l = m
	u = np.zeros(x.size)
	for i in range(x.size):
		n = 0
		for j in range(i-l,i+l+1):
			if j >= 0 and j < x.size:
				n+=1
				u[i] += b[j]
		u[i] = u[i] / float(n)

	return u

@nb.njit(nb.double[:](nb.double[:]))
def guess_baseline(x):
	bg = np.zeros_like(x)
	bg = filters(x,5,x.size//5)
	return bg



if __name__ == "__main__":
	import matplotlib.pyplot as plt
	d,db = fake_data(1000,100.,10.)
	plt.plot(db,'o',markersize=1,alpha=.4,color='k')
	plt.plot(d,color='k',lw=1.,alpha=.5)

	# theta = initial_guess(d)
	# theta = np.array((0,1e-10,.01,.0001))
	# print(theta)
	# for i in range(1000):
	# 	theta,bstar = LDS_round(d,theta)
	# 	print(theta)
	theta,bstar,v,iteration,r2 = LDS(db)
	print((initial_guess(db)))
	print(theta)
	print(iteration)
	plt.fill_between(np.arange(bstar.size),bstar-3.*np.sqrt(v),bstar+3.*np.sqrt(v),color='r',alpha=.3)
	plt.plot(bstar,lw=1,alpha=.5,color='r')

	plt.show()

	# plt.clf()
	# plt.semilogy(r2)
	# plt.show()

	print(theta)
