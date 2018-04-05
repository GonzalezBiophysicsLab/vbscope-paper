#### 1D GMM - EM Max Likelihood

import numpy as np
import numba as nb

from fxns.statistics import p_normal
from fxns.kernel_sample import kernel_sample
from fxns.numba_math import psi,gammaln
from fxns.gmm_related import initialize_params, result_bayesian_gmm

@nb.jit(nb.types.Tuple((nb.float64[:],nb.float64[:],nb.float64[:]))(nb.float64[:],nb.float64[:,:]),nopython=True)
def m_sufficient_statistics(x,r):
	#### M Step
	## Sufficient Statistics
	xbark = np.zeros(r.shape[1])
	sk = np.zeros_like(xbark)

	nk = np.sum(r,axis=0) + 1e-300
	for i in range(nk.size): ## ignore the outlier class
		xbark[i] = 0.
		for j in range(r.shape[0]):
			xbark[i] += r[j,i]*x[j]
		xbark[i] /= nk[i]

		sk[i] = 0.
		for j in range(r.shape[0]):
			sk[i] += r[j,i]*(x[j] - xbark[i])**2.
		sk[i] /= nk[i]
	return nk,xbark,sk

@nb.jit(nb.types.Tuple((nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:]))(nb.float64[:],nb.float64[:,:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:]),nopython=True)
def m_updates(x,r,a0,b0,m0,beta0,alpha0):
	#### M Step
	## Updates
	nk,xbark,sk = m_sufficient_statistics(x,r)

	beta = np.zeros_like(m0)
	m = np.zeros_like(m0)
	a = np.zeros_like(m0)
	b = np.zeros_like(m0)
	alpha = np.zeros_like(m0)

	## Hyperparameters
	for i in range(nk.size):
		beta[i] = beta0[i] + nk[i]
		m[i] = 1./beta[i] *(beta0[i]*m0[i] + nk[i]*xbark[i])
		a[i] = a0[i] + (nk[i]+1.)/2.
		b[i] = .5*(b0[i] + nk[i]*sk[i] + beta0[i]*nk[i]/(beta0[i]+nk[i])*(xbark[i]-m0[i])**2.)
		alpha[i] = alpha0[i] + nk[i]
	return a,b,m,beta,alpha,nk,xbark,sk


@nb.jit(nb.float64[:](nb.float64[:,:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:]),nopython=True)
def calc_lowerbound(r,a,b,m,beta,alpha,nk,xbark,sk,E_lnlam,E_lnpi,a0,b0,m0,beta0,alpha0):

	lt71 = 0.
	lt74 = 0.
	lt77 = 0.
	## Precompute
	lnb0 = a0*np.log(b0) - gammaln(a0)
	lnbk = a*np.log(b) - gammaln(a)
	hk = -lnbk -(a-1.)*E_lnlam + a
	for i in range(m.shape[0]):
		## Data
		lt71 += .5 * nk[i] * (E_lnlam[i] - 1./beta[i] - a[i]/b[i]*(sk[i] - (xbark[i]-m[i])**2.) - np.log(2.*np.pi))

		## Normal Wishart
		lt74 += .5*(np.log(beta0[i]/2./np.pi) + E_lnlam[i] - beta0[i]/beta[i] - beta0[i]*a[i]/b[i]*(m[i]-m0[i])**2.)
		lt74 += lnb0[i] + (a0[i]-1.)*E_lnlam[i] - a[i]*b0[i]/b[i]
		lt77 += .5*E_lnlam[i] + .5*np.log(beta[i]/2.*np.pi) - .5 - hk[i]
		Fgw = lt74 - lt77

	## Dirichlet
	Fa = gammaln(alpha0.sum()) - np.sum(gammaln(alpha0)) + np.sum((alpha0-1.)*E_lnpi)
	Fa -= np.sum((a-1.)*E_lnpi) + gammaln(alpha.sum()) - np.sum(gammaln(alpha))

	## Multinomial
	Fpi = 0.
	for i in range(r.shape[0]):
		for j in range(r.shape[1]):
			Fpi += r[i,j]*(E_lnpi[j] - np.log(r[i,j]))

	ll1 = lt71 + Fgw + Fa + Fpi
	return np.array((ll1,lt71,Fgw,Fa,Fpi))

@nb.jit(nb.types.Tuple((nb.float64[:,:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.int64,nb.float64[:,:]))(nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.int64,nb.float64),nopython=True)
def outer_loop(x,mu,var,ppi,maxiters,threshold):

	## priors - from vbFRET
	beta0 = .25 + np.zeros_like(mu)
	m0 = mu + np.zeros_like(mu)
	a0 = 2.5 + np.zeros_like(mu)
	b0 = 0.01 + np.zeros_like(mu)
	alpha0 = 1. + np.zeros_like(mu)

	# initialize
	prob = p_normal(x,mu,var)
	r = np.zeros_like(prob)
	for i in range(r.shape[0]):
		r[i] = ppi*prob[i]
		r[i] /= np.sum(r[i])

	a,b,m,beta,alpha,nk,xbark,sk = m_updates(x,r,a0,b0,m0,beta0,alpha0)

	E_dld = np.zeros_like(prob)
	E_lnlam = np.zeros_like(mu)
	E_lnpi = np.zeros_like(mu)

	iteration = 0
	ll1 = -np.inf
	ll0 = -np.inf
	ll = np.zeros((maxiters,5))

	while iteration < maxiters:
		# E Step

		for i in range(prob.shape[0]):
			E_dld[i] = 1./beta + a/b*(x[i]-m)**2.
		E_lnlam = psi(a)-np.log(b)
		E_lnpi = psi(alpha)-psi(np.sum(alpha))

		for i in range(r.shape[0]):
			r[i] = np.exp(E_lnpi + .5*E_lnlam - .5*np.log(2.*np.pi) - .5*E_dld[i])
			r[i] /= np.sum(r[i])+1e-10 ## for numerical stability



		ll0 = ll1
		ll[iteration] = calc_lowerbound(r,a,b,m,beta,alpha,nk,xbark,sk,E_lnlam,E_lnpi,a0,b0,m0,beta0,alpha0)
		ll1 = ll[iteration,0]

		## likelihood
		if iteration > 1:
			dl = np.abs((ll1 - ll0)/ll0)
			if dl < threshold or np.isnan(ll1):
				break

		a,b,m,beta,alpha,nk,xbark,sk = m_updates(x,r,a0,b0,m0,beta0,alpha0)


		# mu = m ##
		# var = b/a ##

		if iteration < maxiters:
			iteration += 1
	return r,a,b,m,beta,alpha,E_lnlam,E_lnpi,iteration,ll

def vb_em_gmm(x,nstates,maxiters=1000,threshold=1e-6):
	'''
	Data convention is NxK
	'''

	if x.ndim != 1:
		raise Exception("Input data isn't 1D")

	# mu,var,ppi = initialize_params(x,nstates)
	from ml_em_gmm import ml_em_gmm
	o_ml = ml_em_gmm(x,nstates)

	r,a,b,m,beta,alpha,E_lnlam,E_lnpi,iteration,ll = outer_loop(x,o_ml.mu,o_ml.var,o_ml.ppi,maxiters,threshold)

	result = result_bayesian_gmm(r,a,b,m,beta,alpha,E_lnlam,E_lnpi,ll[:iteration+1],iteration)

	return result
