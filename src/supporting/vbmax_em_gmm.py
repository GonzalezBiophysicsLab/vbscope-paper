#### 1D GMM - EM Max Likelihood

import numpy as np
import numba as nb
import multiprocessing as mp

from .numba_math import psi,gammaln,erf

class result(object):
	def __init__(self, *args):
		self.args = args
class result_bayesian_gmm(result):
	def __init__(self,r,a,b,m,beta,alpha,E_lnlam,E_lnpi,likelihood,iteration):
		self.r = r
		self.a = a
		self.b = b
		self.m = m
		self.beta = beta
		self.alpha = alpha
		self.E_lnlam = E_lnlam
		self.E_lnpi = E_lnpi
		self.iteration = iteration
		self.likelihood = likelihood

		self.mu = m
		self.var = 1./np.exp(E_lnlam)
		self.ppi = self.r.sum(0) / self.r.sum()
		# self.ppi = np.exp(E_lnpi)

	def report(self):
		s = 'VB GMM\n-----------\n'
		s += 'N States: %d\n'%(self.mu.size)
		s += 'ln l: %f\niters: %d\n'%(self.likelihood[-1,0],self.iteration)
		s += 'mu:  %s\n'%(self.mu)
		s += '+/-: %s\n'%(1./np.sqrt(self.beta))
		s += 'var: %s\n'%(self.var)
		s += 'f:   %s\n'%(self.ppi)
		return s

# def initialize_params(x,nstates,flag_kmeans=False):
# 	from .kmeans import kmeans
# 	np.random.seed()
# 	# if not flag_kmeans:
# 		# xx = x[np.random.randint(low=0,high=x.size,size=np.min((xx.size,100)),dtype='i')]
# 		# mu = kernel_sample(x,nstates)
#
# 		# distinv = 1./np.sqrt((x[:,None] - mu[None,:])**2.)
# 		# r = np.exp(-distinv) + .1
# 		# r = r/r.sum(1)[:,None]
# 		# var = np.sum(r*(x[:,None]-mu)**2.,axis=0)/np.sum(r,axis=0) + 1e-300
# 		# ppi = r.mean(0)
# 		# var = np.var(x)/nstates + np.zeros(nstates)
# 		# ppi = 1./nstates + np.zeros(nstates)
#
# 	# else:
# 	if 1:
# 		r,mu,var,ppi = kmeans(x,nstates)
# 		# mu = np.random.normal(loc=mu,scale=np.sqrt(var),size=nstates)
#
# 	return mu,var,ppi

# @nb.njit
def neglnp_maxval(theta,x,n):
	mu,tau = theta
	m = x.size
	y = m*np.log(n)
	y += (n-1.)*np.sum(np.log(.5)+np.log((1.+erf(np.sqrt(tau/2.)*(x-mu)))))
	y += .5*m*(np.log(tau) - np.log(2.*np.pi))
	y -= .5*tau*np.sum((x-mu)**2.)
	return -y
	# return -lnp(x,mu,tau,n)

# @nb.jit
def solve_em(x,n):
	from scipy.optimize import minimize
	guess = np.array((x.mean(),1./x.var()/np.sqrt(n)))
	out = minimize(neglnp_maxval,x0=guess,args=(x,n),method='Nelder-Mead')
	return out

# @nb.njit(nb.double[:](nb.double[:],nb.double,nb.double))
def lnp_normal(x,mu,var):
	y = np.log(2.*np.pi) + np.log(var) + (x-mu)**2./var
	return -.5 * y * (var > 0.)

def p_normal(x,mu,var):
	return np.exp(lnp_normal(x,mu,var))

# @nb.njit(nb.double[:](nb.double[:],nb.int64,nb.double,nb.double))
def lnp_normal_max(x,n,mu,var):
	prec = 1./var
	y = .5+.5*erf((x-mu)*np.sqrt(prec/2.))
	return (n-1.)*np.log(y) + np.log(float(n)) + lnp_normal(x,mu,var)


def vbmax_em_gmm(x, nstates, bg, nmax, initials=None, maxiters=1000, threshold=1e-6, prior_strengths=None, flag_report=True, init_kmeans = False):
	'''
	Data convention is NxK
	bg should be the background normal, not the max val or min val
	'''

	if x.ndim != 1:
		raise Exception("Input data isn't 1D")

	## Priors - beta, a, b, alpha... mu is from GMM
	if prior_strengths is None:
		prior_strengths = np.array((0.25,2.5,.01,1.))

	# if initials is None:
	# 	mu,var,ppi = initialize_params(x,nstates+1, init_kmeans)
	# else:
	# 	mu,var,ppi = initials
	mu = np.linspace(bg[0],np.percentile(x,95),nstates+1)
	var = np.zeros(nstates+1) + np.var(x) - bg[1]
	var[0] = bg[1]
	ppi = np.ones(nstates+1)/(nstates+1.)


	#######
	## priors - from vbFRET
	beta0 = prior_strengths[0] + np.zeros_like(mu)
	m0 = mu.copy() + np.zeros_like(mu)
	a0 = prior_strengths[1] + np.zeros_like(mu)
	b0 = prior_strengths[2] + np.zeros_like(mu)
	alpha0 = prior_strengths[3] + np.zeros_like(mu)

	# initialize
	prob = p_normal(x[:,None],mu[None,:],var[None,:])
	r = ppi[None,:]*prob
	r /= (r.sum(1) + 1e-10)[:,None]
	# a,b,m,beta,alpha,nk,xbark,sk = m_updates(x,r,a0,b0,m0,beta0,alpha0)
	nk = np.sum(r,axis=0)+1e-300
	xbark = np.sum(r*x[:,None],axis=0)/nk
	sk = np.sum(r*(x[:,None]-xbark[None,:])**2.,axis=0)/nk
	beta = beta0+nk
	m = 1./beta * (beta0*m0 + nk*xbark)
	a = a0 + .5*(nk+1.)
	b = .5*(b0 + nk*sk + beta0*nk/(beta0+nk)*(xbark-m0)**2.)
	alpha = alpha0+nk

	iteration = 0
	ll1 = -np.inf
	ll0 = -np.inf
	ll = np.zeros((maxiters))

	while iteration < maxiters:
		#### E Step
		E_dld = 1./beta[None,:] + (a/b)[None,:]*(x[:,None]-m[None,:])**2.
		E_lnlam = psi(a)-np.log(b)
		E_lnpi = psi(alpha)-psi(np.sum(alpha))

		r[:,0] = np.exp(lnp_normal_max(x,nmax,bg[0],bg[1]))
		r[:,1:] = np.exp((E_lnpi[1:] + .5*E_lnlam[1:] - .5*np.log(2.*np.pi))[None,:] - .5*E_dld[:,1:])
		r[x <=bg[0],1:] = 0
		r /= (r.sum(1) + 1e-10)[:,None]

		#### Calculate ELBO
		ll0 = ll1
		lnb0 = a0*np.log(b0) - gammaln(a0)
		lnbk = a*np.log(b) - gammaln(a)
		hk = -lnbk -(a-1.)*E_lnlam + a

		lt71 = np.sum(.5 * nk * (E_lnlam - 1./beta - a/b*(sk - (xbark-m)**2.) - np.log(2.*np.pi)))
		lt74 = .5*(np.log(beta0/2./np.pi) + E_lnlam - beta0/beta - beta0*a/b*(m-m0)**2.)
		lt74 += lnb0 + (a0-1.)*E_lnlam - a*b0/b
		lt77 = .5*E_lnlam + .5*np.log(beta/2.*np.pi) - .5 - hk
		Fgw = np.sum(lt74 - lt77)

		# Dirichlet
		Fa = gammaln(alpha0.sum()) - np.sum(gammaln(alpha0)) + np.sum((alpha0-1.)*E_lnpi)
		Fa -= np.sum((a-1.)*E_lnpi) + gammaln(alpha.sum()) - np.sum(gammaln(alpha))

		# Multinomial
		Fpi = np.sum(r*(E_lnpi[None,:] - np.log(1e-300+r)))

		ll[iteration] = lt71 + Fgw + Fa + Fpi
		ll1 = ll[iteration]

		## Stoping conditions
		if iteration > 2:
			dl = np.abs((ll1 - ll0)/ll0)
			if dl < threshold or np.isnan(ll1):
				break

		#### M Step
		nk = np.sum(r,axis=0)+1e-300
		xbark = np.sum(r*x[:,None],axis=0)/nk
		sk = np.sum(r*(x[:,None]-xbark[None,:])**2.,axis=0)/nk
		beta = beta0+nk
		m = 1./beta * (beta0*m0 + nk*xbark)
		a = a0 + .5*(nk+1.)
		b = .5*(b0 + nk*sk + beta0*nk/(beta0+nk)*(xbark-m0)**2.)
		alpha = alpha0+nk
		if iteration % 10 == 5: ## 5,15,25,...
			out = solve_em(x[r.argmax(1)==0],nmax)
			bg[0] = out.x[0]
			bg[1] = 1./out.x[1]
		m[0] = bg[0]
		b[0] = bg[1]*a[0]

		if iteration < maxiters:
			iteration += 1


	if flag_report:
		result = result_bayesian_gmm(r,a,b,m,beta,alpha,E_lnlam,E_lnpi,ll[:iteration+1],iteration)
		result.prior_strengths = prior_strengths
		print(result.iteration,result.likelihood[-1])
		return result
	return r

def vbmax_em_gmm_parallel(x, nstates, bg, nmax, initials=None, maxiters=1000, threshold=1e-10, nrestarts=1, prior_strengths=None, ncpu=1):

	# if platform != 'win32' and ncpu != 1 and nrestarts != 1:
	# 	pool = mp.Pool(processes = ncpu)
	# 	results = [pool.apply_async(vbmax_em_gmm, args=(x,nstates,bg,nmax,initials,maxiters,threshold,prior_strengths,True)) for i in range(nrestarts)]
	# 	results = [p.get() for p in results]
	# 	pool.close()
	# else:
	# 	results = [vbmax_em_gmm(x,nstates,bg,nmax,initials,maxiters,threshold,prior_strengths,True) for i in range(nrestarts)]
	#
	# try:
	# 	best = np.nanargmax([r.likelihood[-1,0] for r in results])
	# except:
	# 	best = 0
	# return results[best]


	return vbmax_em_gmm(x,nstates,bg,nmax,None,maxiters,threshold,prior_strengths,True, True)
