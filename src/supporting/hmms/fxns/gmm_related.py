import numpy as np
from kernel_sample import kernel_sample

def initialize_params(x,nstates):

	# x = xx[np.random.randint(low=0,high=xx.size,size=np.min((xx.size,100)),dtype='i')]

	mu = kernel_sample(x,nstates)

	distinv = 1./np.sqrt((x[:,None] - mu[None,:])**2.)
	r = distinv/distinv.sum(1)[:,None]
	var = np.sum(r*(x[:,None]-mu)**2.,axis=0)/np.sum(r,axis=0) + 1e-300
	ppi = r.mean(0)

	return mu,var,ppi

class result(object):
	def __init__(self, *args):
		self.args = args

class result_ml_gmm(result):
	def __init__(self,mu,var,r,ppi,likelihood,iteration):
		self.mu = mu
		self.var = var
		self.r = r
		self.ppi = ppi
		self.likelihood = likelihood
		self.iteration = iteration

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
		self.ppi = np.exp(E_lnpi)
