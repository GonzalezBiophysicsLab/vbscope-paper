## Autocorrelations
import numpy as np
import numba as nb
from scipy import stats
import numpy as np

#### testing
# dt = 0.025
# mu = np.array((.1,.3,.6,.9))
# noise = 0.01
# var = np.zeros_like(mu) + noise**2.
# rates = 10. * np.array(((0.,.005,.003,.001),(.001,0.,.002,.002),(.004,.003,.0,.004),(0.002,.003,.001,0.)))
# from scipy.linalg import expm
# q = rates.copy()
# for i in range(q.shape[0]):
# 	q[i,i] = - q[i].sum()
# tmatrix = expm(q*dt)
# tinf = 1000./np.abs(q).min()
# popplot.hmm.t,popplot.hmm.y = gen_mc_acf(1.,popplot.ens.y.size,tmatrix,mu,var,ppi)
# popplot.hmm.t,popplot.hmm.y = gen_mc_acf_q(dt,popplot.ens.y.size,q,mu,var,ppi)
# popplot.hmm.t /= dt

def gen_mc_acf_q(tau,nsteps,q,mu,var,ppi):
	## Using a Q matrix, not a transition probability matrix
	## note..... these are transposed relative to tmatrix out of hmm routines

	nstates,_ = q.shape
	pi0 = np.eye(nstates)

	## get steady state probabilities
	from scipy.linalg import expm
	tinf = 1000./np.abs(q).min()
	pinf = expm(q*tinf)[0]

	## use fluctuations
	mubar =  (pinf*mu).sum()
	mm = mu - mubar

	### expectation here
	E_y0yt = np.zeros(nsteps)
	for k in range(nsteps): # loop over time delay steps
		tp = expm(q*tau*k)
		for i in range(nstates): # loop over initial state
			for j in range(nstates): # loop over final state
				E_y0yt[k] += mm[i]*mm[j] * np.dot(tp.T,pi0[i])[j] * pinf[i]

	## add gaussian noise terms
	for i in range(nstates):
		E_y0yt[0] += var[i]*pinf[i]
	## normalize
	E_y0yt /= E_y0yt[0]

	t = tau*np.arange(nsteps)
	return t,E_y0yt

@nb.njit
def gen_mc_acf(tau,nsteps,tmatrix,mu,var,ppi):
	## Using a transition probability matrix, not a rate matrix, or Q matrix
	## because this comes straight out of an HMM.... so not quick exact

	nstates,_ = tmatrix.shape
	pi0 = np.eye(nstates)

	## get steady state probabilities
	pinf = np.linalg.matrix_power(tmatrix.T,100*int(1./tmatrix.min()))[:,0]

	## use fluctuations
	mubar =  (pinf*mu).sum()
	mm = mu - mubar

	### expectation here
	E_y0yt = np.zeros(nsteps)
	for i in range(nstates): # loop over initial state
		for j in range(nstates): # loop over final state
			for k in range(nsteps): # loop over time delay steps
				## E[y_0*t_t] = \sum_ij m_i * m_j * (A^n \cdot \delta (P_i))_j * P_inf,i
				E_y0yt[k] += mm[i]*mm[j] * (np.dot(np.linalg.matrix_power(tmatrix.T,k),pi0[i])[j]) * pinf[i]

	## add gaussian noise terms
	for i in range(nstates):
		E_y0yt[0] += var[i]*pinf[i]
	## normalize
	E_y0yt /= E_y0yt[0]

	t = tau*np.arange(nsteps)
	return t,E_y0yt

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
		# temp = acorr_counts(dd[i]-abar)
		temp = acorr_counts(dd[i])
		y += temp[0]
		n += temp[1]

	## calculate posterior
	## mn,kn,an,bn
	posterior = np.zeros((4,y.shape[0]))

	# Priors
	a0 = 1.
	k0 = .001
	# m0 = abar**2.
	m0 = 0.
	b0 = 1.

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

	## messing up FFT
	# for i in range(y.size):
	# 	if n[i] == 0:
	# 		posterior[0,i] = np.nan

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
