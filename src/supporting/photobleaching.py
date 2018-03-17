import numpy as np
import numba as nb
from math import lgamma as gammaln

import os

_windows = False
if os.name == 'nt':
	_windows = True

@nb.jit(nb.double(nb.double[:]),nopython=True)
def sufficient_xbar(x):
	n = x.size

	# xbar
	xbar = 0.
	for i in range(n):
		xbar += x[i]
	xbar /= float(n)
	return xbar

@nb.jit(nb.double(nb.double[:],nb.double),nopython=True)
def sufficient_s2(x,m):
	n = x.size

	#s2
	s2 = 0.
	for i in range(n):
		d = x[i]-m
		s2 += d*d
	return s2

################################################################################
### \mathcal{N} with an unknown \mu and an unknown \sigma
################################################################################
@nb.jit(nb.types.Tuple((nb.double,nb.double,nb.double,nb.double))(nb.double[:],nb.double,nb.double,nb.double,nb.double),nopython=True)
def normal_update(x,a0,b0,k0,m0):
	# sufficient statistics
	xbar = sufficient_xbar(x)
	s2 = sufficient_s2(x,xbar)

	# posterior parameters
	n = x.size
	kn = k0 + n
	an = a0 + n/2.
	bn = b0 + .5*s2 + k0*n*(xbar - m0)**2. / (2.*(k0+n))
	mn = (k0*m0 + n*xbar)/kn
	return an,bn,kn,mn

@nb.jit([nb.double(nb.double[:],nb.double,nb.double,nb.double,nb.double)],nopython=True)
def normal_ln_evidence(x,a0,b0,k0,m0):

	n = x.size
	an,bn,kn,mn = normal_update(x,a0,b0,k0,m0)
	ln_evidence = gammaln(an) - gammaln(a0) + a0*np.log(b0) - an*np.log(bn) +.5*np.log(k0) - .5*np.log(kn) - n/2. * np.log(2.*np.pi)

	return ln_evidence

################################################################################
### \mathcal{N} with an known \mu and an unknown \sigma
################################################################################
@nb.jit(nb.types.Tuple((nb.double,nb.double))(nb.double[:],nb.double,nb.double,nb.double),nopython=True)
def normal_mu_update(x,mu,a0,b0):
	# sufficient statistics
	# s2 = np.sum((x-mu)**2.)
	s2 = sufficient_s2(x,mu)

	# posterior parameters
	n = x.size
	an = a0 + n/2.
	bn = b0 + .5*s2
	return an,bn

@nb.jit(nb.double(nb.double[:],nb.double,nb.double,nb.double),nopython=True)
def normal_mu_ln_evidence(x,mu,a0,b0):
	n = x.size

	an,bn = normal_mu_update(x,mu,a0,b0)
	ln_evidence = gammaln(an) - gammaln(a0) + a0*np.log(b0) - an*np.log(bn) - n/2. * np.log(2.*np.pi)

	return ln_evidence

################################################################################
### Photobleaching model - start w/ N(\mu) go to N(0) at time t
################################################################################

@nb.jit(nb.double[:](nb.double[:],nb.double,nb.double,nb.double,nb.double),nopython=True)
def ln_likelihood(d,a0,b0,k0,m0):
	lnl = np.zeros_like(d)

	lnl[0] = normal_mu_ln_evidence(d,0.,a0,b0)

	for i in range(1,d.shape[0]-1):
		lnl[i] = normal_ln_evidence(d[:i],a0,b0,k0,m0) + normal_mu_ln_evidence(d[i:],0.,a0,b0)
	lnl[-1] = normal_ln_evidence(d,a0,b0,k0,m0)
	return lnl

@nb.jit(nb.double(nb.double[:],nb.double,nb.double,nb.double,nb.double),nopython=True)
def ln_evidence(d,a0,b0,k0,m0):

	lnl  = ln_likelihood(d,a0,b0,k0,m0)
	# uniform priors for t
	lmax = lnl.max()
	ev = np.log(np.sum(np.exp(lnl-lmax)))+lmax
	return ev

@nb.jit(nb.double(nb.double[:]))
def ln_bayes_factor(d):
	a0 = 1.
	b0 = 1.
	k0 = 1.
	m0 = 1000.
	return ln_evidence(d,a0,b0) - normal_ln_evidence(d,a0,b0,k0,m0)

@nb.jit(nb.double[:](nb.double[:],nb.double,nb.double,nb.double,nb.double,nb.double),nopython=True)
def posterior(d,k,a0,b0,k0,m0):
	t = np.arange(d.size)
	lnp = ln_likelihood(d,a0,b0,k0,m0) + np.log(k) - k*t
	return lnp


################################################################################
### Single Step Model
################################################################################

@nb.jit(nb.int64(nb.double[:]),nopython=True)
def get_point_pbtime(d):
	a0 = 1.
	b0 = 1.
	k0 = 1.
	m0 = 1000.
	lnl = ln_likelihood(d,a0,b0,k0,m0)
	# for i in range(lnl.shape[0]):
	# 	if np.isnan(lnl[i]):
	# 		lnl[i] = -np.inf
	# pbt = lnl.argmax()
	pbt = np.argmax(lnl)
	return pbt

@nb.jit(nb.double(nb.double[:]),nopython=True)
def get_expectation_pbtime(d):
	a0 = 1.
	b0 = 1.
	k0 = 1.
	m0 = 1000.
	lnl = ln_likelihood(d,a0,b0,k0,m0)
	t = np.arange(lnl.size)
	lmax = np.max(lnl)
	p = np.exp(lnl-lmax)
	psum = np.sum(p)
	pbt = np.sum(p*t)/psum
	return pbt

if _windows:
	@nb.jit(nb.types.Tuple((nb.double,nb.int64[:]))(nb.double[:,:]),nopython=True)
	def pb_ensemble(d):
		'''
		Input:
			* `d` is a np.ndarray of shape (N,T) of the input data
		Output:
			* `e_k` is the expectation of the photobleaching time rate constant
			* `pbt` is a np.ndarray of shape (N) with the photobleaching time
		'''
		pbt = np.zeros(d.shape[0],dtype=nb.int64)
		for i in range(d.shape[0]):
			pbt[i] = get_expectation_pbtime(d[i])
		e_k = (1.+pbt.size)/(1.+np.sum(pbt))
		for i in range(d.shape[0]):
			pbt[i] = np.argmax(posterior(d[i],e_k))
		return e_k,pbt

else:
	@nb.jit(nb.types.Tuple((nb.double,nb.int64[:]))(nb.double[:,:]),nopython=True,parallel=True)
	def pb_ensemble(d):
		'''
		Input:
			* `d` is a np.ndarray of shape (N,T) of the input data
		Output:
			* `e_k` is the expectation of the photobleaching time rate constant
			* `pbt` is a np.ndarray of shape (N) with the photobleaching time
		'''
		a0 = 1.
		b0 = 1.
		k0 = 1.
		m0 = 1000.

		pbt = np.zeros(d.shape[0],dtype=nb.int64)
		for i in nb.prange(d.shape[0]):
			pbt[i] = get_expectation_pbtime(d[i])
		e_k = (1.+pbt.size)/(1.+np.sum(pbt))
		for i in nb.prange(d.shape[0]):
			pbt[i] = np.argmax(posterior(d[i],e_k,a0,b0,k0,m0))
		return e_k,pbt

@nb.jit(nb.double[:](nb.double[:,:]),nopython=True)
def pb_snr(d):
	pbt = pb_ensemble(d)[1]
	snrr = np.zeros(d.shape[0],dtype=nb.double)
	for i in range(snrr.size):
		if pbt[i] > 5:
			if pbt[i] < d.shape[1] - 5:
				snrr[i] = (np.mean(d[i,:pbt[i]]) - np.mean(d[i,pbt[i]:])) / np.std(d[i,:pbt[i]])
			else:
				snrr[i] = np.mean(d[i])/np.std(d[i])
		else:
			snrr[i] = 0.
	return snrr

################################################################################
## Check to see if a signal is zero
################################################################################
@nb.jit(nb.double[:](nb.double[:,:]),nopython=True)
def model_comparison_signal(x):
	out = np.zeros(x.shape[0],dtype=nb.double)
	a0 = 1.
	b0 = 1.
	k0 = 1.
	m0 = 1000.
	for i in range(out.size):
		lnp_m2 = normal_mu_ln_evidence(x[i],0.,a0,b0)
		lnp_m1 = normal_ln_evidence(x[i],a0,b0,k0,m0)
		p = 1./(1.+np.exp(lnp_m2-lnp_m1))
		out[i] = p
	return out

################################################################################
######################### Jaewook's variance > 1 model #########################
################################################################################
### calc_pb_time will return a list of integers of the photobleaching frame.
### remove_pb will return a copy of the data with NaN after the photobleachings.
################################################################################
################################################################################

@nb.jit(nb.boolean[:](nb.double[:],nb.double),nopython=True)
def sliding_var_greater(d,l):
	## Calculate windowed variance for each datapoint
	## l is the window size
	out = np.zeros(d.size,dtype=nb.boolean)
	for i in range(d.size):
		## Deal with the start and end of the array
		xmin = np.max(np.array([0,i-l]))
		xmax = np.min(np.array([i+l,d.size-1]))
		## Calculate that variance
		v = np.var(d[xmin:xmax+1])
		## Find the first time it's above 1.0
		if v > 1.0:
			out[i] = True
	return out

@nb.jit(nb.int64(nb.double[:],nb.double),nopython=True)
def first_var_greater(d,l):
	## Calculate windowed variance for each datapoint
	## l is the window size
	for i in range(d.size):
		## Deal with the start and end of the array
		xmin = np.max(np.array([0,i-l]))
		xmax = np.min(np.array([i+l,d.size-1]))
		## Calculate that variance
		v = np.var(d[xmin:xmax+1])
		## Find the first time it's above 1.0
		if v > 1.0:
			return i
	## If it's never > 1.0
	return d.size


if _windows:
	@nb.jit(nb.int64[:](nb.double[:,:],nb.int64),nopython=True)
	def calc_pb_time(d,l):
		## Find the first point where the variance is greater than 1.0 for all traces
		t = np.zeros((d.shape[0]),dtype=nb.int64)
		for i in range(t.size):
			t[i] = first_var_greater(d[i],l)
		return t
### This problem is embarassingly parallel. Process each trace individual, in parallel
else:
	@nb.jit(nb.int64[:](nb.double[:,:],nb.int64),nopython=True,parallel=True)
	def calc_pb_time(d,l):
		## Find the first point where the variance is greater than 1.0 for all traces
		t = np.zeros((d.shape[0]),dtype=nb.int64)
		for i in nb.prange(t.size):
			t[i] = first_var_greater(d[i],l)
		return t

@nb.jit(nb.double[:,:](nb.double[:,:]),nopython=True)
def remove_pb_first(d):
	### d should be an np.ndarray of shape (N,T) where N is the number of traces, and T is the length of each trace

	## Find first time when var(x_{i-2}, ..., x{i+2}) > 1.0
	## This will include photobleaching and Cy3 blinks
	t = calc_pb_time(d,2)

	## Turn all points after the first bleach/blink point into NaNs
	out = np.copy(d)
	for i in range(d.shape[0]):
		out[i,t[i]:] = np.nan
	return out

@nb.jit(nb.double[:,:](nb.double[:,:]),nopython=True)
def remove_pb_all(d):
	### d should be an np.ndarray of shape (N,T) where N is the number of traces, and T is the length of each trace

	## Turn all points after the first bleach/blink point into NaNs
	out = np.copy(d)
	for i in range(d.shape[0]):
		bad = sliding_var_greater(d[i],5)
		out[i][bad] = np.nan
	return out
