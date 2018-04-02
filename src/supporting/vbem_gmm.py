import numpy as np
np.seterr(all="ignore")
from scipy.special import psi,gammaln
import normal_minmax_dist as nmd

class theta():
	def __init__(self,nstates):
		self.m = np.zeros(nstates)
		self.beta= np.ones(nstates)
		self.a = np.ones(nstates)
		self.b = np.ones(nstates)
		self.u = np.ones(nstates)
		self.update_expectations()

	def update_expectations(self):
		# Wikipeia, Bishop Appendix, Winn Appendix
		self.e_lam = self.a/self.b
		self.e_mu = self.m
		self.e_mu2 = self.m**2. + 1./self.beta
		self.e_lnlam = psi(self.a) - np.log(self.b)
		self.e_lnpi = psi(self.u) -psi(self.u.sum())

	def copy(self):
		t = theta(self.m.size)
		t.m = self.m.copy()
		t.beta = self.beta.copy()
		t.a = self.a.copy()
		t.b = self.b.copy()
		t.u = self.u.copy()
		t.update_expectations()
		return t

def initialize_priors(x,k):
	prior  = theta(k)

	from scipy.stats import gaussian_kde
	kernel = gaussian_kde(x)
	mu = kernel.resample(k).flatten()
	mu.sort()
	# from sklearn.cluster import k_means
	# mu = k_means(x.reshape((x.size,1)),k)[0].flatten()
	# mu.sort()

	# xmin = np.percentile(x,.00001)
	# xmax = np.percentile(x,99.99999)
	# np.random.seed()
	# mu = np.random.uniform(xmin,xmax,size=k)
	# mu.sort()

	# vbFRET priors are alpha = 1, a = 2.5, b = 0.01, beta = 0.25
	# mu, .1, .005,.25
	delta = 2.**16. # 16-bit camera

	prior.m = mu
	prior.a = np.zeros_like(mu)+1.
	prior.b = np.zeros_like(mu)+1./(36./(delta**2.))
	prior.beta = np.zeros_like(mu)+(36./(delta**2.))*100.

	return prior

class background():
	def __init__(self,n,mu,var):
		self.n = n
		self.mu = mu
		self.var = var

		self.e_max_var = nmd._estimate_var(self.n)
		self.e_max_m = nmd._estimate_mu(self.n,self.var) + self.mu

	def lnprob(self,x):
		return nmd.lnp_normal_max(x,self.n,self.mu,self.var)

def initialize_priors_bg(x,k,bg):
	# Only use points that are above background
	p_bg = np.exp(bg.lnprob(x)) # prob of being max-val background
	cut = (x > bg.e_max_m)*(p_bg < p_bg.max()*.01) # when drops < 0.1% and greater than mean
	if cut.sum() < 1:
		cut = np.isfinite(x)
	prior = initialize_priors(x[cut],k-1)
	prior.m = np.append(prior.m[0] - 1e-6,prior.m)
	prior.beta = np.append(prior.beta[0] - 1e-6,prior.beta)
	prior.a = np.append(prior.a[0] - 1e-6,prior.a)
	prior.b = np.append(prior.b[0] - 1e-6,prior.b)
	prior.u = np.append(prior.u[0] - 1e-6,prior.u)

	# prior = initialize_priors(x,k)

	return prior

class vbem_gmm():
	def __init__(self,x,nstates,bg=None,prior=None):
		'''
		Make bg a background class object to make the 0^th class a max-value normal distribution set by bg
		Keeping bg as None is just a regular VBEM GMM
		'''

		self.k = nstates
		self.x = x
		self.n = x.size

		self._bg_flag = False if bg is None else True
		if self._bg_flag: self.background = bg

		# Initialize Priors
		if prior is None and self._bg_flag:
			self.k += 1
			self.prior = initialize_priors_bg(self.x,self.k,self.background)

		elif prior is None and not self._bg_flag:
			self.prior = initialize_priors(self.x,self.k)
		else:
			self.prior = prior

		# Update and initialize
		self.post = self.prior.copy()
		self.post.update_expectations()
		self.update_responsibilities()

		self.lowerbound = -np.inf

		self.threshold = 1e-10
		self.maxiters = 1000
		self._debug = False


	def update_stats(self): #ignore bg
		### Bishop pg 477
		nk = self.r.sum(0) + 1e-300
		xbar = np.sum(self.r*self.x[:,None],axis=0) / nk
		sk = np.sum(self.r*(self.x[:,None] - xbar[None,:])**2.,axis=0) / nk
		self.stats = np.array((nk,xbar,sk))

	def update_posterior(self): # ignore bg except in Dirichlet
		### Bishop pg 478, Winn pg 29 (1.88-1.92)

		# \mu
		self.post.beta = self.prior.beta + self.post.e_lam * self.stats[0]
		self.post.m = self.prior.beta*self.prior.m + self.post.e_lam * self.stats[0] * self.stats[1]
		self.post.m /= self.post.beta

		# \lambda
		self.post.a = self.prior.a + .5 * self.stats[0]
		self.post.b = self.prior.b + .5 * np.sum(self.r * (self.x[:,None]**2. - 2.*self.x[:,None]*self.post.e_mu[None,:] + self.post.e_mu2[None,:]), axis=0)

		# \pi
		self.post.u = self.prior.u + self.stats[0]

		# Hacks for max-value distribution
		if self._bg_flag:
			self.post.m[0] = self.background.mu
			self.post.beta[0] = 1.0
			self.post.a[0] = 1.0
			self.post.b[0] = self.background.var

		# Update Expectations
		self.post.update_expectations()

	def update_responsibilities(self):
		# Winn 1.93, Bishop pg 479

		# Regular classes - normal distributions
		self.e_lnp_x =  self.x[:,None]*self.post.e_mu[None,:]
		self.e_lnp_x += -.5*(self.post.e_mu2[None,:] + self.x[:,None]**2.)
		self.e_lnp_x *= self.post.e_lam[None,:]
		self.e_lnp_x += -.5*np.log(2.*np.pi)
		self.e_lnp_x += .5*self.post.e_lnlam[None,:]

		# Background class - max-val normal distribution
		if self._bg_flag:
			self.e_lnp_x[:,0] = self.background.lnprob(self.x)

		# Dirichlet contribution
		self.r = (psi(self.post.u) - psi(self.post.u.sum()))[None,:] + self.e_lnp_x

		# Log prob to prob
		self.r -= self.r.max(1)[:,None]
		self.r = np.exp(self.r)
		self.r += 1e-300
		self.r /= self.r.sum(1)[:,None]

		# Calculate class probabilities
		self.pi = self.r.sum(0)/self.r.sum()

	def update(self):
		self.update_stats()
		self.update_posterior()
		self.update_responsibilities()
		self.calc_lowerbound()

	def dkl_gauss(self): # ignore bg
		y  = self.prior.beta * (self.post.m - self.prior.m)**2.
		y += self.prior.beta / self.post.beta - 1.
		y += - np.log(self.prior.beta) - np.log(self.post.beta)
		return 0.5* y

	def dkl_gamma(self): # ignore bg
		y  =  (self.post.a - self.prior.a)*psi(self.post.a)
		y += -gammaln(self.post.a) + gammaln(self.prior.a)
		y +=  self.prior.a*(np.log(self.post.b) - np.log(self.prior.b))
		y +=  self.post.a*(self.prior.b - self.post.b)/self.post.b
		return y

	def dkl_dirichlet(self):
		y  =  gammaln(self.post.u.sum())  - gammaln(self.post.u).sum()
		y += -gammaln(self.prior.u.sum()) + gammaln(self.prior.u).sum()
		y += np.sum((self.post.u-self.prior.u)*(psi(self.post.u) - psi(self.post.u.sum())))
		return y

	def calc_lowerbound(self):
		skip = 1 if self._bg_flag else 0
		self.lowerbound = (self.r * (self.e_lnp_x + self.post.e_lnpi[None,:] - np.log(self.r)))[:,skip:].sum() # gaussian nodes contribution
		if self._bg_flag: self.lowerbound += (self.r[:,0]*self.background.lnprob(self.x)).sum() # background contribution
		self.lowerbound -= self.dkl_gauss()[skip:].sum() # only nodes
		self.lowerbound -= self.dkl_gamma()[skip:].sum() # only nodes
		self.lowerbound -= self.dkl_dirichlet() # all

	def run(self):
		l0 = self.lowerbound

		for iterations in range(self.maxiters):
			self.update()

			if self._debug:
				print "%05d %.20f %.20f %.20f"%(iterations,l0,self.lowerbound,np.abs(l0 - self.lowerbound)/np.abs(self.lowerbound))

			if np.abs(l0 - self.lowerbound)/np.abs(self.lowerbound) < self.threshold:
				break

			l0 = self.lowerbound

def _run(a):
	a.run()
	return a

def robust_vbem(x,nstates,bg=None,nrestarts=7,nthreads=None,maxiters=1000):
	import multiprocessing as mp

	vbems = [vbem_gmm(x,nstates,bg)]

	if nrestarts > 1:
		for i in range(nrestarts - 1):
			vbems.append(vbem_gmm(x,nstates,bg))

	for i in range(len(vbems)):
		vbems[i].maxiters = maxiters

	if not nthreads is None and nrestarts > 1:
		pool = mp.Pool(nthreads)
		vbems = pool.map(_run,vbems)
		pool.close()

	else:
		vbems = map(_run,vbems)

	lbs = np.array([vv.lowerbound for vv in vbems])
	return vbems[lbs.argmax()]
