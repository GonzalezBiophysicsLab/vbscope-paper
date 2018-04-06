#### 1D HMM - EM Max Likelihood

import numpy as np
import numba as nb
from sys import platform
import multiprocessing as mp

from fxns.statistics import p_normal,dkl_dirichlet
from fxns.kernel_sample import kernel_sample
from fxns.numba_math import psi,gammaln
from fxns.gmm_related import initialize_params, result_bayesian_gmm
from fxns.hmm_related import forward_backward,viterbi,result_bayesian_hmm,initialize_tmatrix

@nb.jit(nb.types.Tuple((nb.float64[:],nb.float64[:],nb.float64[:]))(nb.float64[:],nb.float64[:,:]),nopython=True)
def m_sufficient_statistics(x,r):
	#### M Step
	## Sufficient Statistics
	xbark = np.zeros(r.shape[1])
	sk = np.zeros_like(xbark)

	nk = np.sum(r,axis=0) + 1e-10
	for i in range(nk.size):
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


@nb.jit(nb.float64[:](nb.float64[:,:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:,:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:,:],nb.float64),nopython=True)
def calc_lowerbound(r,a,b,m,beta,pik,tm,nk,xbark,sk,E_lnlam,E_lnpi,a0,b0,m0,beta0,pi0,tm0,lnz):

	lt74 = 0.
	lt77 = 0.
	## Precompute
	lnb0 = a0*np.log(b0) - gammaln(a0)
	lnbk = a*np.log(b) - gammaln(a)
	hk = -lnbk -(a-1.)*E_lnlam + a
	for i in range(m.shape[0]):

		## Normal Wishart
		lt74 += .5*(np.log(beta0[i]/2./np.pi) + E_lnlam[i] - beta0[i]/beta[i] - beta0[i]*a[i]/b[i]*(m[i]-m0[i])**2.)
		lt74 += lnb0[i] + (a0[i]-1.)*E_lnlam[i] - a[i]*b0[i]/b[i]
		lt77 += .5*E_lnlam[i] + .5*np.log(beta[i]/2.*np.pi) - .5 - hk[i]
		Fgw = lt74 - lt77

	## Starting point
	Fpi = - dkl_dirichlet(pik,pi0)

	## Transition matrix
	Ftm = 0.
	for i in range(tm.shape[0]):
		Ftm -= dkl_dirichlet(tm[i],tm0[i])

	ll1 = lnz + Fgw + Fpi + Ftm
	return np.array((ll1,lnz,Fgw,Fpi,Ftm))

@nb.jit(nb.types.Tuple((nb.float64[:,:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:,:],nb.float64[:],nb.float64[:],nb.float64[:,:],nb.int64,nb.float64[:,:]))(nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:,:],nb.int64,nb.float64,nb.float64[:]),nopython=True)
def outer_loop(x,mu,var,tm,maxiters,threshold,prior_strengths):

	## priors - from vbFRET
	beta0 = prior_strengths[0] + np.zeros_like(mu)
	m0 = mu + np.zeros_like(mu)
	a0 = prior_strengths[1] + np.zeros_like(mu)
	b0 = prior_strengths[2] + np.zeros_like(mu)
	pi0 = prior_strengths[3] + np.zeros_like(mu)
	tm0 = prior_strengths[4] + np.zeros_like(tm)

	# initialize
	prob = p_normal(x,mu,var)
	r = np.zeros_like(prob)
	for i in range(r.shape[0]):
		r[i] = prob[i]
		r[i] /= np.sum(r[i]) + 1e-10 ## for stability

	a,b,m,beta,pik,nk,xbark,sk = m_updates(x,r,a0,b0,m0,beta0,pi0)

	E_lntm = np.zeros_like(tm)
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
		E_lnpi = psi(pik)-psi(np.sum(pik))
		for i in range(E_lntm.shape[0]):
			E_lntm[i] = psi(tm[i])-psi(np.sum(tm[i]))


		for i in range(prob.shape[1]):
			prob[:,i] = (2.*np.pi)**-.5 * np.exp(-.5*(E_dld[:,i] - E_lnlam[i]))
		r, xi, lnz = forward_backward(prob, np.exp(E_lntm), np.exp(E_lnpi))

		ll0 = ll1
		ll[iteration] = calc_lowerbound(r,a,b,m,beta,pik,tm,nk,xbark,sk,E_lnlam,E_lnpi,a0,b0,m0,beta0,pi0,tm0,lnz)
		ll1 = ll[iteration,0]

		## likelihood
		if iteration > 1:
			dl = np.abs((ll1 - ll0)/ll0)
			if dl < threshold or np.isnan(ll1):
				break

		a,b,m,beta,pik,nk,xbark,sk = m_updates(x,r,a0,b0,m0,beta0,pi0)
		pik = pi0 + r[0]
		tm  = tm0 + xi.sum(0)

		if iteration < maxiters:
			iteration += 1
	return r,a,b,m,beta,pik,tm,E_lnlam,E_lnpi,E_lntm,iteration,ll

def vb_em_hmm(x,nstates,maxiters=1000,threshold=1e-10,prior_strengths=None):
	'''
	Data convention is NxK
	'''

	if x.ndim != 1:
		raise Exception("Input data isn't 1D")

	## Priors - beta, a, b, pi, alpha... mu is from GMM
	if prior_strengths is None:
		prior_strengths = np.array((0.25,2.5,.01,1.,1.))

	# from ml_em_gmm import ml_em_gmm
	# o = ml_em_gmm(x,nstates+1)
	# mu = o.mu[:-1]
	# var = o.var[:-1]
	# ppi = o.ppi[:-1]
	# ppi /= ppi.sum() ## ignore outliers

	mu,var,ppi = initialize_params(x,nstates)
	tmatrix = initialize_tmatrix(nstates)

	r,a,b,m,beta,pi,tmatrix,E_lnlam,E_lnpi,E_lntm,iteration,likelihood = outer_loop(x,mu,var,tmatrix,maxiters,threshold,prior_strengths)

	result = result_bayesian_hmm(r,a,b,m,beta,pi,tmatrix,E_lnlam,E_lnpi,E_lntm,likelihood[:iteration+1],iteration)
	result.viterbi = viterbi(x,result.mu,result.var,result.tmatrix,result.ppi)

	return result


def vb_em_hmm_parallel(x,nstates,maxiters=1000,threshold=1e-10,nrestarts=1,prior_strengths=None,ncpu=1):

	if platform != 'win32' and ncpu != 1 and nrestarts != 1:
		pool = mp.Pool(processes = ncpu)
		results = [pool.apply_async(vb_em_hmm, args=(x,nstates,maxiters,threshold,prior_strengths)) for i in xrange(nrestarts)]
		results = [p.get() for p in results]
		pool.close()
	else:
		results = [vb_em_hmm(x,nstates,maxiters,threshold,prior_strengths) for i in xrange(nrestarts)]

	try:
		best = np.nanargmax([r.likelihood[-1,0] for r in results])
	except:
		best = 0
	return results[best]
