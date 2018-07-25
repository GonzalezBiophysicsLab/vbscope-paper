## Autocorrelations
import numpy as np
import numba as nb
from scipy import stats
import numpy as np

def gen_acf(tau,nsteps,tmatrix,mu,ppi=None):
	## Using a transition probability matrix, not a rate matrix, or Q matrix
	## because this comes straight out of an HMM

	nstates,_ = tmatrix.shape
	pi0 = np.eye(nstates)

	## get steady state probabilities
	ninf = nsteps*100
	pinf = np.dot(np.linalg.matrix_power(tmatrix.T,ninf),pi0[0][:,None]) ## start anywhere...

	## use fluctuations
	mubar =  (pinf.flatten()*mu).sum()
	mm = mu - mubar

	n = np.arange(nsteps)
	t = tau*n

	E_y0yt = np.zeros((nstates,nstates,nsteps))
	for i in range(nstates): # loop over initial state
		for j in range(nstates): # loop over final state
			for k in range(len(n)): # loop over time delay steps
				## E[y_0*t_t] = \sum_ij m_i * m_j * (A^n \cdot \delta (P_i))_j * P_inf,i
				E_y0yt[i,j,k] = mm[i]*mm[j] * (np.dot(np.linalg.matrix_power(tmatrix.T,n[k]),pi0[i])[j]) * pinf[i]

	## take expectation value
	z = E_y0yt.sum((0,1))

	## normalize
	z /= z[0]

	return t,z

@nb.jit(nopython=True)
def acorr(d):
	## calculate the autocorrelation function
	c = acorr_counts(d)
	return c[0]/c[1]

@nb.jit(nopython=True)
def acorr_counts(d):
	## calc sum and counts for single trace
	out = np.zeros((2,d.size))
	for i in range(d.size): ## delay
		for j in range(0,d.size-i):
			a = d[j]*d[i+j]
			if not np.isnan(a):
				out[0,i] += a
				out[1,i] += 1.0
	return out

@nb.jit(nopython=True)
def ensemble_bayes_acorr(dd):
	## dd shape is NxT with NaN pads on left and right for bad data
	N,T = dd.shape
	y = np.zeros((T))
	n = np.zeros((T))

	nm1 = 0.
	nm2 = 0.
	for i in range(N):
		for j in range(T):
			if not np.isnan(dd[i,j]):
				nm1 +=dd[i,j]
				nm2 += 1.
	abar = nm1/nm2

	for i in range(N):
		temp = acorr_counts(dd[i]-abar)
		y += temp[0]
		n += temp[1]

	## calculate posterior
	## mn,kn,an,bn
	posterior = np.zeros((4,y.shape[0]))

	# Priors
	a0 = .001
	k0 = .001
	m0 = 0.
	b0 = .5

	ybar = y/n

	an = a0 + n/2.
	kn = k0 + n
	mn = (k0*m0 + y)/kn
	bn = b0 + k0*n*(ybar-m0)**2. / (2.*kn)

	for k in range(N):
		d = dd[k]
		for i in range(d.size):
			for j in range(0,d.size-i):
				yy = d[j]*d[i+j]
				if not np.isnan(yy): ## these datapoints do
					bn[i] += .5*(yy - ybar[i])**2.
	posterior[0] = mn
	posterior[1] = kn
	posterior[2] = an
	posterior[3] = bn

	return posterior

	# return posterior

@nb.jit(nopython=True)
def single_bayes_acorr(d):
	out = acorr_counts(d)

	## Calculate the posterior of the autocorrelation function
	## Normal Gamma Parameterization
	# y is sums
	# n is counts
	y = out[0]
	n = out[1]

	## mn,kn,an,bn
	posterior = np.zeros((4,y.size))

	# Priors
	a0 = .001
	k0 = .001
	m0 = 0.5
	b0 = .5

	# Evaluate Posteriors
	ybar = y/n

	an = a0 + n/2.
	kn = k0 + n
	mn = (k0*m0 + y)/kn
	bn = b0 + k0*n*(ybar-m0)**2. / (2.*kn)

	for i in range(d.size):
		for j in range(0,d.size-i):
			yy = d[j]*d[i+j]
			if not np.isnan(yy): ## these datapoints do
				bn[i] += .5*(yy - ybar[i])**2.
	posterior[0] = mn
	posterior[1] = kn
	posterior[2] = an
	posterior[3] = bn
	return posterior

def credible_interval(posterior,p=.95):
	## generate credible interval lower and upper lines from a posterior
	dp = (1.-p)/2. ## ie 2.5 if p = 95%
	ps = np.array([dp,1.-dp]) ## ie 2.5 to 97.5 if p = 95%

	mn,kn,an,bn = posterior
	nu = 2.*an
	sig = np.sqrt(bn/(an*kn))

	# Use Student's T-distribution PPF from scipy.stats.t
	ci = stats.t.ppf(ps[:,None], nu[None,:], loc=mn[None,:], scale=sig[None,:])
	return ci

def plot_bayes_acorr(d,axis=None,color='blue',normalize = True):
	import matplotlib.pyplot as plt
	if axis is None:
		f,axis = plt.subplots(1)

	posterior = ensemble_bayes_acorr(d)
	ci = credible_interval(posterior)
	t = np.arange(posterior[0].size)

	if normalize:
		norm = posterior[0][0]
	else:
		norm = 1.

	axis.fill_between(t, ci[0]/norm, ci[1]/norm, alpha=.3, color=color)
	axis.plot(t, posterior[0]/norm, color=color, lw=1., alpha=.9)

## tests
if __name__ == '__main__':
	from scipy.ndimage import median_filter
	import matplotlib.pyplot as plt

	n = 1000
	dd = np.zeros((10,n))
	for i in range(dd.shape[0]):
		d = np.random.normal(size=n).cumsum()
		d = d - median_filter(d,n/100)
		l = np.random.randint(low=0,high=n)
		d[:l] += np.nan
		h = np.random.randint(low=l, high=n)
		# print l,h
		d[h:] += np.nan
		dd[i] = d - np.nanmean(d)

	plot_bayes_acorr(dd)
	for i in range(dd.shape[0]):
		y = acorr(dd[i])
		plt.plot(y/y[0],'r',lw=.5,alpha=.1)

	# plt.xscale('log')
	plt.xlim(0.0,100.)
	plt.ylim(-1,1)
	plt.show()
