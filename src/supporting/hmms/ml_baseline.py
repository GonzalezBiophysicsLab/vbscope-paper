import numpy as np
import numba as nb

@nb.njit
def tridiagonal_solver(a,b,c,d):
	cp = np.zeros_like(c)
	dp = np.zeros_like(d)
	x = np.zeros_like(d)

	cp[0] = c[0]/b[0]
	dp[0] = d[0]/b[0]
	for i in range(1,d.size):
		cp[i] = c[i]/(b[i]-a[i]*cp[i-1])
		dp[i] = (d[i]-a[i]*dp[i-1])/(b[i]-a[i]*cp[i-1])

	x[-1] = dp[-1]
	for j in range(x.size-1):
		i = x.size - 2 -j
		x[i] = dp[i] - cp[i]*x[i+1]

	return x

@nb.njit(nb.double[:](nb.double,nb.double[:]))
def solve_baseline(r2,d):
	"""
	Expectation Step - calculate best baseline
	"""
	tdm_a = np.zeros(d.size) - 1.
	tdm_b = np.zeros(d.size) + (r2 + 2.)
	tdm_c = np.zeros(d.size) - 1.
	tdm_b[0] = 1. + r2
	tdm_b[-1] = 1. + r2
	tdm_d = r2 * d
	bstar = tridiagonal_solver(tdm_a,tdm_b,tdm_c,tdm_d)
	return bstar

def optimal_r2_rootfinder(r2,baseline,sk2):
	"""
	Maximization Step - returns R2
	"""

	from scipy.optimize import root
	def minr2fxn(r2,baseline,sk2):
		if r2 > 1e-1 or r2 < 1e-15:
			return np.inf
		b = np.sum((baseline[1:]-baseline[:-1])**2.)
		dqdr2 = b/(2.*sk2*r2**2.)
		dqdr2 -= baseline.size*(1.+(2.+r2)/np.sqrt(4.*r2+r2**2.))/(2.+r2+np.sqrt(4*r2+r2**2.))
		return dqdr2
	r2 = root(minr2fxn, x0 = np.array((r2)), args=(baseline,sk2))
	return r2.x[0]

@nb.njit(nb.double(nb.double[:]))
def optimal_r2(x):
	T = x.size
	d = np.sum((x[1:]-x[:-1])**2.)
	return np.real((2.*(2.+d))/(3.*(-1.+T**2.))-(2.**(1./3.)*(-4.*(2.+d)**2.-3.*d*(8.+d)*(-1.+T**2.)))/(3.*(-1.+T**2.)*(128.-96.*d+24.*d**2.-2.*d**3.+288.*d*T**2.-36.*d**2.*T**2.+18.*d**3.*T**2.+108.*d**2.*T**4.+np.sqrt((128.-96.*d+24.*d**2.-2.*d**3.+288.*d*T**2.-36.*d**2.*T**2.+18.*d**3.*T**2.+108.*d**2.*T**4.)**2.+4.*(-4.*(2.+d)**2.-3.*d*(8.+d)*(-1.+T**2.))**3.))**(1./3.))+(1./(3.*2.**(1./3.)*(-1.+T**2.)))*((128.-96.*d+24.*d**2.-2.*d**3.+288.*d*T**2.-36.*d**2.*T**2.+18.*d**3.*T**2.+108.*d**2.*T**4.+np.sqrt((128.-96.*d+24.*d**2.-2.*d**3.+288.*d*T**2.-36.*d**2.*T**2.+18.*d**3.*T**2.+108.*d**2.*T**4.)**2.+4.*(-4.*(2.+d)**2.-3.*d*(8.+d)*(-1.+T**2.))**3.))**(1./3.)))

@nb.njit(nb.types.Tuple((nb.double[:],nb.double,nb.double))(nb.double[:]))
def estimate_baseline(x):
	r2 = 1.
	maxiters = 1000
	threshold = 1e-5
	record = np.zeros(maxiters)
	iteration = 0
	while iteration < maxiters:
		bstar = solve_baseline(r2,x)
		r2 = optimal_r2(bstar)
		vn = np.var(x-bstar)
		vb = r2*vn
		record[iteration] = r2
		if iteration > 1:
			if np.abs(record[iteration] - record[iteration-1])/record[iteration] < threshold:
				break
		iteration += 1
	return bstar,vn,vb


if __name__ == "__main__":
	np.random.seed(666)
	d = (np.random.normal(size=10000)*.2).cumsum() + np.random.normal(size=10000)*np.sqrt(.5)

	bstar,vn,vb = estimate_baseline(d)
	print((vn,vb))

	import matplotlib.pyplot as plt
	plt.plot(d)
	plt.plot(bstar)
	plt.show()
