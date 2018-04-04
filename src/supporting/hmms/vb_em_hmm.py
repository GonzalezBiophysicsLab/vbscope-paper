#### 1D HMM - EM Max Likelihood

import numpy as np
import numba as nb

from fxns.statistics import p_normal,dkl_dirichlet
from fxns.kernel_sample import kernel_sample
from fxns.numba_math import psi,gammaln
from fxns.gmm_related import initialize_params, result_bayesian_gmm
from fxns.hmm_related import forward_backward,viterbi,result_bayesian_hmm,initialize_tmatrix

@nb.jit(nb.types.Tuple((nb.float64[:],nb.float64[:],nb.float64[:]))(nb.float64[:],nb.float64[:,:]),nopython=True,cache=True)
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

@nb.jit(nb.types.Tuple((nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:]))(nb.float64[:],nb.float64[:,:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:]),nopython=True,cache=True)
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


@nb.jit(nb.float64[:](nb.float64[:,:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:,:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:,:],nb.float64),nopython=True,cache=True)
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

@nb.jit(nb.types.Tuple((nb.float64[:,:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:,:],nb.float64[:],nb.float64[:],nb.float64[:,:],nb.int64,nb.float64[:,:]))(nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:,:],nb.int64,nb.float64),nopython=True,cache=True)
def outer_loop(x,mu,var,tm,maxiters,threshold):

	## priors - from vbFRET
	beta0 = .25 + np.zeros_like(mu)
	m0 = mu + np.zeros_like(mu)
	a0 = 2.5 + np.zeros_like(mu)
	b0 = 0.01 + np.zeros_like(mu)
	pi0 = 1. + np.zeros_like(mu)
	tm0 = np.ones_like(tm)

	# initialize
	prob = p_normal(x,mu,var)
	r = np.zeros_like(prob)
	for i in range(r.shape[0]):
		r[i] = prob[i]
		r[i] /= np.sum(r[i])

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

		# mu = m ##
		# var = b/a ##

		if iteration < maxiters:
			iteration += 1
	return r,a,b,m,beta,pik,tm,E_lnlam,E_lnpi,E_lntm,iteration,ll

def vb_em_hmm(x,nstates,maxiters=1000,threshold=1e-10):
	'''
	Data convention is NxK
	'''

	if x.ndim != 1:
		raise Exception("Input data isn't 1D")

	from ml_em_gmm import ml_em_gmm
	o = ml_em_gmm(d,nstates)
	mu = o.mu
	var = o.var
	ppi = o.ppi

	tmatrix = initialize_tmatrix(nstates)

	r,a,b,m,beta,pi,tmatrix,E_lnlam,E_lnpi,E_lntm,iteration,likelihood = outer_loop(x,mu,var,tmatrix,maxiters,threshold)

	result = result_bayesian_hmm(r,a,b,m,beta,pi,tmatrix,E_lnlam,E_lnpi,E_lntm,likelihood[:iteration+1],iteration)


	return result



# ### Examples
from fxns.fake_data import fake_data
t,d = fake_data()

# nstates = 2
lls = []
for nstates in range(1,11):
	o = vb_em_hmm(d,nstates)
	print nstates
	lls.append(o.likelihood)
l = [ll[-1,0] for ll in lls]
import matplotlib.pyplot as plt
plt.figure()
plt.plot(range(1,11),l,'-o')
plt.show()
#
# print o.tmatrix
# print o.mu
# print o.var**.5
# print o.r.shape
#
# import matplotlib.pyplot as plt
cm = plt.cm.jet
v = viterbi(d,o.mu,o.var,o.tmatrix,o.ppi)
f,a = plt.subplots(1)
for i in range(nstates):
	c = cm(float(i)/nstates)
	# print i,c
	xcut = v==i
	a.plot(t[xcut],d[xcut],'o',color=c,alpha=.5)
	# plt.axhline(y=o.mu[i], color=c)
a.plot(t,o.mu[v])

# a[1] = plt.plot(o.likelihood[:,0],'k')
plt.show()
