import numpy as np
from solid.kmeans import kmeans
from scipy.special import psi,gammaln
from solid import normal_minmax_dist as nmd

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

def initialize_priors(x,k,init_kmeans=True):
	prior  = theta(k)

	if init_kmeans:
		km = kmeans(x,k,nrestarts=10)
		xsort = np.argsort(km.mu[:,0])
		mu = km.mu[:,0][xsort]
		var = km.var[:,0,0][xsort] + 1e-300
		var = var.max()
		var = np.var(x)
		# if np.any(~np.isfinite(var)): var = (np.abs(x.max()-x.min()))
		# var = (np.abs(x.max()-x.min()))**1.

	else:
		xmin = x.min()
		xmax = x.max()
		# mu = np.random.rand(k)*(xmax-xmin) + xmin
		# mu.sort() # keep states ordered by mu
		dx = (xmax-xmin)/k
		np.random.seed()
		mu = np.linspace(xmin,xmax,k) + np.random.randn(k)*dx**.5

		if mu.size == 1:
			var = np.var(x)
		else:
			var = np.var(mu)
		# var = (dx)**2.

	# var = (np.abs(x.max()-x.min()))#np.var(x)

	# Setup priors
	prior.m = mu

	# prior.beta *= 1./np.abs(prior.m).max()#**.5
	prior.beta *= 1./var**.75
	# prior.beta *= 1./var

	prior.a *= 1.

	# prior.b *= var**.5
	prior.b *= var**.75
	# prior.b *= np.var(x)
	# prior.b *= (np.abs(x.max()-x.min()))

	return prior

class background():
	def __init__(self,n,mu,var):
		self.n = n
		self.mu = mu
		self.var = var

		self.e_max_var_max = nmd._estimate_var(self.n)
		self.e_max_m = nmd._estimate_mu(self.n,self.var) + self.mu

	def lnprob(self,x):
		return nmd.lnp_normal_max(x,self.n,self.mu,self.var)

def initialize_priors_bg(x,k,bg,init_kmeans=True):
	# Only use points that are above background
	p_bg = np.exp(bg.lnprob(x)) # prob of being max-val background
	cut = (x > bg.e_max_m)*(p_bg < p_bg.max()*.05) # when drops < 5% and greater than mean
	prior = initialize_priors(x[cut],k-1,init_kmeans=init_kmeans)
	prior.m = np.append(prior.m[0] - 1e-20,prior.m)
	prior.beta = np.append(prior.beta[0] - 1e-20,prior.beta)
	prior.a = np.append(prior.a[0] - 1e-20,prior.a)
	prior.b = np.append(prior.b[0] - 1e-20,prior.b)
	prior.u = np.append(prior.u[0] - 1e-20,prior.u)

	# prior = initialize_priors(x,k,init_kmeans=init_kmeans)

	return prior

class vbem_gmm():
	def __init__(self,x,nstates,bg=None,prior=None,init_kmeans=True):
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
			self.prior = initialize_priors_bg(self.x,self.k,self.background,init_kmeans=init_kmeans)

		elif prior is None and not self._bg_flag:
			self.prior = initialize_priors(self.x,self.k,init_kmeans=init_kmeans)
		else:
			self.prior = prior

		# Update and initialize
		self.post = self.prior.copy()
		self.post.update_expectations()
		self.update_responsibilities()

		self.lowerbound = -np.inf

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
		if self._bg_flag: self.e_lnp_x[:,0] = self.background.lnprob(self.x)

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

	def run(self,maxiters=1000,threshold=1e-6,debug=False):
		l0 = self.lowerbound

		for iterations in range(maxiters):
			self.update()

			if debug:
				print "%05d %.20f %.20f %.20f"%(iterations,l0,self.lowerbound,np.abs(l0 - self.lowerbound)/np.abs(self.lowerbound))

			if np.abs(l0 - self.lowerbound)/np.abs(self.lowerbound) < threshold:
				break

			l0 = self.lowerbound


def _run(a):
	a.run(threshold=1e-10,maxiters=1000)
	return a

def robust_vbem(x,nstates,bg=None,nrestarts=7,nthreads=None):
	import multiprocessing as mp


	vbems = [vbem_gmm(x,nstates,bg,init_kmeans=1)]
	if nrestarts > 1:
		for i in range(nrestarts - 1):
			vbems.append(vbem_gmm(x,nstates,bg,init_kmeans=False))

	if not nthreads is None:
		pool = mp.Pool(nthreads)
		vbems = pool.map(_run,vbems)
		pool.close()

	else:
		vbems = map(_run,vbems)

	lbs = np.array([vv.lowerbound for vv in vbems])
	print lbs
	print lbs.argmax()
	pool.close()
	return vbems[lbs.argmax()]




#### Testing
if 1:
	import matplotlib.pyplot as plt
	from solid import normal_minmax_dist as nmd

	n = np.array([200,400,300,100,])*10
	mu = np.array((500.,600.,400.,1250.))*.5
	nsearch = 9
	bg = np.random.randn(nsearch,500)*20. + 200.
	bg *= .5
	bgmin = nmd.estimate_from_min(bg.min(0),nsearch)
	print bg.max(0).mean(),np.var(bg.max(0))

	x = np.concatenate([np.random.poisson(lam=mu[i],size=n[i]) for i in range(n.size)]).astype('f')
	x += np.random.randn(x.size)
	x = np.concatenate([x,bg.max(0)])

	print x.size

	nstates = 5

	# if 0:
	# 	o = kmeans(x,nstates)
	# 	hy,hx = plt.hist(x,bins=80,alpha=.5,normed=True)[:2]
	# 	xx = np.linspace(0,x.max()*1.1,100000)
	# 	[plt.plot(xx,nmd.p_normal(xx,o.mu[i],o.var[i,0,0])*o.pi[i],'k') for i in range(nstates)]
	# 	plt.plot(xx,nmd.p_normal_max(xx,nsearch,*bgmin)*bg.shape[1]/x.size)
	# 	plt.show()

	bgbg = background(nsearch,*bgmin)
	a = vbem_gmm(x,nstates,bg=bgbg,init_kmeans=0)
	# a = vbem_normal_bg(x,4,nsearch,bgmin[0],bgmin[1],init_kmeans=False)
	# a = [vbem_normal_bg(x,i,nsearch,bgmin[0],bgmin[1],init_kmeans=True) for i in range(1,10)]
	a.run(threshold=1e-10,debug=1)

	# a = vbem_gmm(x,nstates,bg=bgbg,init_kmeans=False)
	# a.run()
	print a.lowerbound
	# aa = []
	# for nstates in [1,2,3,4,5,6,7,8,9]:
	# 	a = robust_vbem(x,nstates,bg=bgbg,nthreads=4)
	# 	aa.append(a)
	# lbs = np.array([aaa.lowerbound for aaa in aa])
	# plt.figure()
	# plt.plot([1,2,3,4,5,6,7,8,9],lbs)
	# plt.show()
	# plt.figure()
	# b = a.prior
	# b.m = a.post.m
	# b.b = a.post.b/a.post.a
	# b.beta *= 1./np.abs(x.max()-x.min())**.5#1./np.abs(b.m.max())**.5
	# a = vbem_gmm(x,nstates,bg=bgbg,prior=b)
	# a.run(threshold=1e-10,debug=True)

	# for i,j in zip(a.pi,a.post.m):
		# print i,j

	skip = 1
	cut = range(a.prior.m.size)
	# cut = np.nonzero(~a.cut())[0]

	if 1:
		hy,hx = plt.hist(x,bins=np.max((80,int(x.size**.5))),alpha=.5,normed=True,histtype='stepfilled')[:2]
		xx = np.linspace(0,x.max()*1.1,10000)

		[plt.plot(xx,nmd.p_normal(xx,a.prior.m[i],1./a.prior.beta[i]),'g',ls='--') for i in cut]
		[plt.plot(xx,nmd.p_normal(xx,a.prior.m[i],a.prior.b[i]/a.prior.a[i]),'k',ls='--') for i in cut]

		[plt.plot(xx,nmd.p_normal(xx,a.post.m[i],a.post.b[i]/a.post.a[i])*a.pi[i],'k',ls='-') for i in cut[1:]]
		plt.plot(xx,np.array([nmd.p_normal(xx,a.post.m[i],a.post.b[i]/a.post.a[i])*a.pi[i] for i in cut[1:]]).sum(0),color='blue')

		plt.plot(xx,nmd.p_normal_max(xx,nsearch,*bgmin)*a.pi[0],color='r')

		plt.yscale('log')
		plt.ylim(1e-5,1)


		print a.prior.m
		print a.post.m
		print a.pi
		plt.show()


	# [aa.run(debug=False,threshold=1e-6,maxiters=100) for aa in a]
	# lbs = [aa.lowerbound for aa in a ]

	# print lbs
	# a = [vbem_normal(x,nstates,init_kmeans=True)]
	# a[0].run(debug=True,threshold=1e-16)
	# # for i in range(500):
	# # 	l0 = a[0].lowerbound
	# # 	a[0].update()
	# # 	print i,l0,a[0].lowerbound
	# # a = [run_vbem(x,2,init_kmeans=True)]
	# # for i in range(4):
	# # 	a.append(run_vbem(x,2,init_kmeans=False))
	#
	# cm = plt.cm.spectral
	# for i in range(len(a)):
	# 	aa = a[i]
	# 	aa.pi = aa.r.sum(0)/aa.r.sum()
	# 	[plt.plot(xx,p_normal(xx,aa.post.m[j],aa.post.b[j]/aa.post.a[j])*aa.pi[j],color=cm(float(i)/float(len(a)))) for j in range(nstates)]
	# # plt.xlim(hx.min(),hx.max())
	# plt.show()
	#
	# print [aa.lowerbound for aa in a]
	# print [aa.post.m for aa in a]
	#
