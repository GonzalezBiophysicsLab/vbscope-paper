#### Simultaneous 1D VBEM HMM

import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from time import time

np.seterr(divide='ignore')

from forward_backward import numba_forward_backward as forward_backward
from forward_backward import numba_viterbi as viterbi
from maths import E_ln_p_x_z, E_ln_pi, E_ln_A, p_normal
from maths import update_tmatrix, update_rho, update_normals, dkl_normalgamma, dkl_dirichlet, dkl_tmatrix

class results_hmm:
	def __init__(self, *args):
		self.args = args

	def gen_report(self,tau = 1.):
		try:
			report = "%s\nHMM - k = %d, iter= %d, lowerbound=%f"%(self.lowerbound,self.m.size,self.iterations,self.lowerbound)
			report += '\n    f: %s'%(self.ppi)
			report += '\n    m: %s'%(self.m)
			report += '\nm_sig: %s'%(1./np.sqrt(self.beta))
			report += '\n  sig: %s'%((self.b/self.a)**.5)
			rates = -np.log(1.-self.Astar)/tau
			for i in range(rates.shape[0]):
				rates[i,i] = 0.
			report += '\n   k:\n'
			report += '%s'%(rates)
		except:
			report = "Bad Report"
		return report

def initialize_priors(data,nstates,flag_vbfret=True,flag_custom=False,flag_user=False):
	# NxTxKxD
	y = np.concatenate(data)
	npoints = y.size

	xmin = np.percentile(y,.01)
	xmax = np.percentile(y,99.99)
	np.random.seed()
	# # m = np.random.uniform(xmin,xmax,size=nstates)
	m = np.linspace(xmin,xmax,nstates)
	m += np.random.normal(size=nstates)*(xmax-xmin)/6
	# ## Pick randomly in region defined by min and max
	# # m = np.array([np.random.uniform(xmin[i],xmax[i],nstates) for i in range(ndim)]).T
	m.sort()
	# print m


	# dist = np.sqrt(np.square(y[:,None] - m[None,:]))
	# gamma = (dist == dist.min(1)[:,None]).astype('double') + 0.1
	# gamma /= gamma.sum(1)[:,None]
	# rho = gamma.sum(0)
	# rho /= rho.sum()
	# rho += 1.
	rho = np.ones(nstates)

	if flag_custom:
		from scipy.stats import gaussian_kde
		kernel = gaussian_kde(y.flatten())
		m = kernel.resample(nstates).flatten()
		m.sort()

		# alpha = np.zeros((nstates,nstates)) + .1 + np.identity(nstates)*1.#1.#10.
		# alpha = np.zeros((nstates,nstates)) + np.identity(nstates)
		# #.1,.005,.25
		# a = np.zeros(nstates) + 2.5#1.
		# b = np.zeros(nstates) + (xmax-xmin)**2./36.#(xmax-xmin)**2./36.
		# beta = np.zeros(nstates) + nstates**2. * 36. / (xmax-xmin)**2.#nstates**2. * 36. / (xmax-xmin)**2.

		alpha = np.ones((nstates,nstates))
		a = np.zeros(nstates) + 2.5
		b = np.zeros(nstates) + .01
		beta = np.zeros(nstates) + .25

	elif flag_vbfret:
		## vbFRET!!
		# vbFRET priors are alpha = 1, a = 2.5, b = 0.01, beta = 0.25
		alpha = np.ones((nstates,nstates))
		a = np.zeros(nstates) + 2.5
		b = np.zeros(nstates) + 0.01
		beta = np.zeros(nstates) + 0.25

	return [m,beta,a,b,alpha,rho]

def simultaneous_vbem_hmm(data,nstates,prior,verbose=False,maxiterations=1000,threshold=1e-10,consensus=True):
	'''
	Format is # NxTxKxD
	Data should be a list of nmol np.ndarray(npoints) trjectories. If only one trace, try [y]
	'''
	# maxiterations = 1000
	# threshold = 1e-16

	# Check Data
	# if y.ndim != 3:
	# 	raise Exception("Data should be Molecules x Time x Dimensionality")
	nmol = len(data)
	# if not wiener_smooth is False:
	# 	from scipy.signal import wiener
	# 	data = [wiener(dd) for dd in data]
	flaty = np.concatenate(data)

	# Parse Prior
	m_0,beta_0,a_0,b_0,alpha_0,rho_0 = prior

	# Initialize Posterior
	m_n = m_0.copy()
	beta_n = beta_0.copy()
	a_n = a_0.copy()
	b_n = b_0.copy()
	alpha_n = alpha_0.copy()
	rho_n = rho_0.copy()

	lowerbounds = np.array([-np.nan])

	# Iteration loop
	for iteration in xrange(maxiterations):

		# E-Step: Log Expectations
		ln_p_x_z = [E_ln_p_x_z(y,m_n,beta_n,a_n,b_n) for y in data]
		ln_A = E_ln_A(alpha_n)
		ln_pi = E_ln_pi(rho_n)

		# Forward-Backward for multiple trajectories
		gamma = []
		xi = []
		ln_z = []
		times = []
		for i in xrange(nmol):
			t0 = time()
			_g,_x,_l = forward_backward(np.exp(ln_p_x_z[i]),np.exp(ln_A),np.exp(ln_pi))
			t1 = time()
			times.append(t1-t0)
			gamma.append(_g)
			xi.append(_x)
			ln_z.append(_l)

		# Calculate Lowerbound
		lowerbound = np.sum(ln_z)
		dkl = dkl_normalgamma(m_n,beta_n,a_n,b_n,m_0,beta_0,a_0,b_0)
		dkl += dkl_tmatrix(alpha_n,alpha_0)
		dkl += dkl_dirichlet(rho_n,rho_0)
		lowerbound -= dkl
		if verbose:
			print iteration,lowerbound,np.mean(times)

		# Convergence Bookkeeping
		delta_change = np.abs((lowerbound - lowerbounds[-1])/lowerbounds[-1])
		lowerbounds = np.append(lowerbounds,lowerbound)
		if iteration == 0:
			lowerbounds = lowerbounds[1:]
		else:
			if iteration > 2 and ((delta_change < threshold)):# or (lowerbounds[-1] == lowerbounds[-3])):
				converged = True
				break

		# M-Step: Calculate Hyperparameters that Maximize Distribution
		m_n, beta_n, a_n, b_n  =  update_normals(flaty, np.concatenate(gamma,axis=0), m_0, beta_0, a_0, b_0)
		alpha_n = alpha_0 + np.sum(np.concatenate(xi,axis=0),axis=0)
		rho_n = rho_0 + np.sum([g[0] for g in gamma],axis=0)

	# Collect and Format Output
	result = results_hmm()
	result.iterations = iteration
	result.nstates = nstates
	result.nmol = nmol
	result.gamma = gamma
	result.lowerbounds = lowerbounds
	result.lowerbound = lowerbound

	# Hyperparameters
	result.m = m_n
	result.beta = beta_n
	result.a = a_n
	result.b = b_n
	result.alpha = alpha_n
	result.rho = rho_n

	# result._m = m_0
	# result._beta = beta_0
	# result._a = a_0
	# result._b = b_0
	# result._alpha = alpha_0
	# result._rho = rho_0

	# Get Dirichlet Means
	result.Astar = alpha_n / alpha_n.sum(1)[:,None]
	result.pistar = rho_n / rho_n.sum()

	# Calculate the Viterbi Paths
	result.viterbi = []
	for i in xrange(nmol):
		p_x_z = p_normal(data[i],m_n,b_n/a_n)
		v = viterbi(p_x_z,result.Astar,result.pistar).astype('i')
		# v = viterbi(np.exp(ln_p_x_z[i]),np.exp(ln_A[i]),np.exp(ln_pi[i]))
		result.viterbi.append(v)
	result.ln_p_x_z = ln_p_x_z

	result.ppi = np.sum([result.gamma[i].sum(0) for i in range(len(result.gamma))],axis=0)
	result.ppi /= np.nansum(result.ppi)

	return result

def hmm_with_restarts(y,nstates,priors,nrestarts=8,prefs=None):
	import multiprocessing as mp

	if not prefs is None:
		wiener_smooth = prefs['hmm_wiener_smooth']
		threshold = prefs['hmm_threshold']
		maxiters = prefs['hmm_max_iters']
		ncpu = prefs['ncpu']
	else:
		wiener_smooth = False
		threshold = 1e-10
		maxiters = 1000
		ncpu = np.min((nrestarts,mp.cpu_count()))

	if not wiener_smooth is False:
		from scipy.signal import wiener
		y = [wiener(dd) for dd in y]

	from sys import platform
	if platform != 'win32':
		pool = mp.Pool(processes = np.min((ncpu,mp.cpu_count())))
		results = [pool.apply_async(simultaneous_vbem_hmm, args=(y,nstates,priors[i],False,maxiters,threshold)) for i in xrange(nrestarts)]
		results = [p.get() for p in results]
		pool.close()
	else:
		results = [simultaneous_vbem_hmm(y,nstates,priors[i],False,maxiters,threshold) for i in xrange(nrestarts)]

	lbs = [results[i].lowerbounds[-1] for i in xrange(nrestarts)]
	iters = [results[i].iterations for i in xrange(nrestarts)]

	best = np.nanargmax(lbs)
	return results[best],lbs
