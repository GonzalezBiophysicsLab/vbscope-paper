import numpy as np
from .kernel_sample import kernel_sample
from .kmeans import kmeans
import time

def initialize_params(x,nstates,flag_kmeans=False):
	np.random.seed()
	if not flag_kmeans:
		# xx = x[np.random.randint(low=0,high=x.size,size=np.min((xx.size,100)),dtype='i')]
		mu = kernel_sample(x,nstates)

		# distinv = 1./np.sqrt((x[:,None] - mu[None,:])**2.)
		# r = np.exp(-distinv) + .1
		# r = r/r.sum(1)[:,None]
		# var = np.sum(r*(x[:,None]-mu)**2.,axis=0)/np.sum(r,axis=0) + 1e-300
		# ppi = r.mean(0)
		var = np.var(x)/nstates + np.zeros(nstates)
		ppi = 1./nstates + np.zeros(nstates)

	else:
		r,mu,var,ppi = kmeans(x,nstates)
		# mu = np.random.normal(loc=mu,scale=np.sqrt(var),size=nstates)

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

	def report(self):
		s = 'ML GMM\n-----------\n'
		s += 'N States: %d\n'%(self.mu.size)
		s += 'ln l: %f\niters: %d\n'%(self.likelihood,self.iteration)
		s += 'mu:  %s\n'%(self.mu)
		s += 'var: %s\n'%(self.var)
		s += 'f:   %s\n'%(self.ppi)
		return s


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
