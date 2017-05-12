import numpy as np
from kmeans import kmeans
from scipy.special import psi,gammaln
from normal_dist import ln_maxval_normal,p_normal
from poisson_dist import ln_maxval_poisson,p_poisson

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

class background():
	def __init__(self,n,mu,var):
		self.n = n
		self.mu = mu
		self.var = var
	
	def lnprob(self,x):
		return ln_maxval_normal(x,self.n,self.mu,self.var)
		# return p_normal(x,self.mu,self.var)

class vbem_normal_bg():
	def __init__(self,x,nstates,bg_n,bg_mu,bg_var,init_kmeans=True):
		self.k = nstates
		self.x = x
		self.n = x.size
		
		self.background = background(bg_n,bg_mu,bg_var)
		p_bg = self.background.lnprob(x)
		p_bg = np.exp(p_bg - p_bg.max())
		
		# Initialize
		self.prior = theta(self.k + 1)
		self.post  = theta(self.k + 1)
		
		# Calculate for priors
		cut = p_bg < 1e-6
		if cut.sum() < nstates: 
			cut = p_bg < 1. # give up!
			
		if init_kmeans:
			km = kmeans(x[cut],nstates,nrestarts=10)
			xsort = np.argsort(km.mu[:,0])
			mu = km.mu[:,0][xsort]
			var = km.var[:,0,0][xsort] + 1e-300
			
			print mu
		else:
			xmin = x[cut].min()
			xmax = x[cut].max()
			mu = np.random.rand(nstates)*(xmax-xmin) + xmin
			mu.sort() # keep states ordered by mu
			var = np.var(x[cut])
		
		# Setup priors
		self.prior.m[1:] = mu
		self.post.m[1:]  = mu
		self.prior.beta[1:] /= var/100.
		self.post.beta[1:]  /= var/100.
		self.prior.b[1:] *= var/100.
		self.post.b[1:] *= var/100.
		
		# self.prior.m[0] = self.background.mu
		# self.prior.beta = 10.
		# self.prior.a = 10.
		# self.prior.b = self.background.var
		
		# Update and initialize
		self.post.update_expectations()
		self.update_probability()
		
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
		
		# Update Expectations
		self.post.update_expectations()
	
	def update_probability(self):
		# Winn 1.93, Bishop pg 479

		self.e_lnp_x =  self.x[:,None]*self.post.e_mu[None,:]
		self.e_lnp_x += -.5*(self.post.e_mu2[None,:] + self.x[:,None]**2.)
		self.e_lnp_x *= self.post.e_lam[None,:]
		self.e_lnp_x += -.5*np.log(2.*np.pi)
		self.e_lnp_x += .5*self.post.e_lnlam[None,:]
		
		self.e_lnp_x[:,0] = self.background.lnprob(self.x)
		
		self.r = (psi(self.post.u) - psi(self.post.u.sum()))[None,:] + self.e_lnp_x
		
		self.r -= self.r.max(1)[:,None]
		self.r = np.exp(self.r)
		self.r += 1e-300
		self.r /= self.r.sum(1)[:,None]
		
		self.pi = self.r.sum(0)/self.r.sum()
		
	def update(self):
		self.update_stats()
		self.update_posterior()
		self.update_probability()
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
		self.lowerbound = (self.r * (self.e_lnp_x + self.post.e_lnpi[None,:] - np.log(self.r)))[:,1:].sum() # gaussian nodes contribution
		# self.lowerbound += (self.r[:,0]*np.log(self.background.prob(self.x))).sum() # background contribution
		self.lowerbound += (self.r[:,0]*self.background.lnprob(self.x)).sum() # background contribution
		self.lowerbound -= self.dkl_gauss()[1:].sum() # only nodes
		self.lowerbound -= self.dkl_gamma()[1:].sum() # only nodes
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
		# print iterations,self.lowerbound
		



#### Testing
if 0:
	import matplotlib.pyplot as plt
	from normal_dist import p_normal

	n = np.array([200,400,300,100,])*10
	mu = np.array((500.,750.,400.,800.))
	x = np.concatenate([np.random.poisson(lam=mu[i],size=n[i]) for i in range(n.size)])

	nstates=8
	# o = kmeans(x,nstates)


	# a = vmp_normal(x,nstates)
	# a.prior.m = o.mu[:,0]
	# a.prior.beta *= 0.01
	# a.prior.a = np.ones(nstates)
	# a.prior.b = o.var[:,0,0]/100.
	#
	# for i in range(100):
	# 	a.update()
	# print a.post.__dict__


	# hy,hx = plt.hist(x,bins=80,alpha=.5,normed=True)[:2]
	xx = np.linspace(0,x.max()*1.1,10000)
	# [plt.plot(xx,p_normal(xx,o.mu[i],o.var[i,0,0])*o.pi[i],'k') for i in range(nstates)]

	a = [vbem_normal(x,i,init_kmeans=True) for i in range(1,10)]
	[aa.run(debug=False,threshold=1e-6,maxiters=100) for aa in a]
	lbs = [aa.lowerbound for aa in a ]
	
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
