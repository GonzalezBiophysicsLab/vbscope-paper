import numpy as np
from kmeans import kmeans
from scipy.special import psi,gammaln

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

class vbem_normal():
	def __init__(self,x,nstates,init_kmeans=True):
		self.k = nstates
		self.x = x
		self.n = x.size
		
		# Initialize
		self.prior = theta(self.k)
		self.post = theta(self.k)
		
		# Calculate for priors
		if init_kmeans:
			km = kmeans(x,nstates,nrestarts=10)
			xsort = np.argsort(km.mu[:,0])
			mu = km.mu[:,0][xsort]
			var = km.var[:,0,0][xsort] + 1e-300
		else:
			mu = np.random.rand(nstates)*(x.max()-x.min()) + x.min()
			mu.sort() # keep states ordered by mu
			var = np.var(x)	
		
		# Setup priors
		self.prior.m = mu
		self.post.m  = mu
		self.prior.beta /= var
		self.post.beta  /= var
		# self.prior.b *= var
		# self.post.b *= var
		# self.prior.a *= var
		# self.post.a *= var
		
		# Update
		self.post.update_expectations()
		
		# Initialize
		# self.r = np.array([np.random.dirichlet(self.prior.u) for _ in range(self.n)])
		self.update_probability()
		
		self.lowerbound = -np.inf
	
	def update_stats(self):
		### Bishop pg 477
		nk = self.r.sum(0) + 1e-300
		xbar = np.sum(self.r*self.x[:,None],axis=0) / nk
		sk = np.sum(self.r*(self.x[:,None] - xbar[None,:])**2.,axis=0) / nk
		self.stats = np.array((nk,xbar,sk))
	
	def update_posterior(self):
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
		self.e_lnp_x = .5 * (self.post.e_lnlam[None,:] - self.post.e_lam[None,:]*(self.x[:,None]**2. - 2.*self.x[:,None]*self.post.e_mu[None,:] + self.post.e_mu2[None,:]))
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
	
	def dkl_gauss(self):
		y  = self.prior.beta * (self.post.m - self.prior.m)**2. 
		y += self.prior.beta / self.post.beta - 1.
		y += - np.log(self.prior.beta) - np.log(self.post.beta)
		return 0.5* y
	
	def dkl_gamma(self):
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
		self.lowerbound = 0. # Need to put this in still... <lnp(joint)>

		self.lowerbound = np.sum(self.r * (self.e_lnp_x + self.post.e_lnpi[None,:] - np.log(self.r)))
		self.lowerbound -= self.dkl_gauss().sum()
		self.lowerbound -= self.dkl_gamma().sum()
		self.lowerbound -= self.dkl_dirichlet()
	
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
if 1:
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
