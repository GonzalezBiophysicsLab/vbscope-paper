import numpy as np
from kmeans import kmeans
from scipy.special import psi,gammaln

class theta():
	def __init__(self,nstates):
		self.k = nstates
		self.m = np.zeros(self.k)
		self.beta= np.ones(self.k)
		self.a = np.ones(self.k)
		self.b = np.ones(self.k)

class vmp_normal():
	def __init__(self,x,nstates):
		self.k = nstates
		self.x = x
		self.prior = theta(self.k)
		self.post = theta(self.k)
		
		self.e_mu = np.zeros((2,self.k))
		self.e_lam = np.zeros((2,self.k))
		
		self.r = np.array([np.random.dirichlet(np.ones(self.k)) for _ in range(self.x.size)])
		
		self.lowerbound = -np.inf
	
	def update_r(self):
		p = self.calc_lnp_xi(self.x)
		p -= p.max(1)[:,None]
		p = np.exp(p)
		
		self.r = p/(p.sum(1)[:,None])
	
	def update(self):

		self.update_mu()
		self.update_lam()
		
		self.update_r()
		self.calc_lowerbound()
	
	### See Winn, J. (Thesis) - 2.25-2.28 (pg. 39)
	
	def update_e_mu(self):
		# <\mu>
		self.e_mu[0] = self.post.m
		# <\mu^2>
		self.e_mu[1] = self.post.m**2. + 1./self.post.beta
	
	def update_e_lam(self):
		# <\lambda>
		self.e_lam[0] = self.post.a/self.post.b
		# <ln \lambda>
		self.e_lam[1] = psi(self.post.a) - np.log(self.post.b)
	
	def update_mu(self):
		phi_0 =  self.prior.beta * self.prior.m 
		phi_0 += (self.e_lam[0][None,:]*self.x[:,None]*self.r).sum(0)
		phi_1 =  -.5*self.prior.beta 
		phi_1 += (-.5*self.e_lam[0][None,:]*self.r).sum(0)
		
		self.post.m = -.5 * phi_0/phi_1
		self.post.beta = -2.*phi_1
		
		self.update_e_mu()
		self.update_lnl_mu()
		
	def update_lam(self):
		phi_0 = -self.prior.b 
		phi_0 += - .5 * np.sum(((self.x[:,None]*self.r)**2. - 2.*self.x[:,None]*self.r*self.e_mu[0][None,:] + (self.e_mu[1])[None,:]),axis=0) 
		phi_1 = self.prior.a - 1. + .5*self.r.sum(0)
		
		self.post.a = phi_1 + 1.
		self.post.b = -phi_0
		
		self.update_e_lam()
		self.update_lnl_lam()
		
	### See Winn, J. (Thesis) - 2.37-2.39 (pg. 41)
	def update_lnl_mu(self):
		y1 = (self.prior.beta*self.prior.m - self.post.beta*self.post.m) * self.e_mu[0]
		y2 = (-self.prior.beta/2. + self.post.beta/2.)*self.e_mu[1]
		y3 = .5*(np.log(self.post.beta) - self.post.beta*self.post.m**2. - np.log(self.prior.beta) + self.prior.beta*self.prior.m**2.)
		self.lnl_mu = y1+y2+y3
	
	def update_lnl_lam(self):
		y1 = (-self.prior.b + self.post.b)*self.e_lam[0]
		y2 = (self.post.a - self.prior.a)*self.e_lam[1]
		y3 = self.post.a*np.log(self.post.b) - gammaln(self.post.a) - self.prior.a*np.log(self.prior.b) + gammaln(self.prior.a)
		self.lnl_lam = y1+y2+y3
	
	def calc_lnp_xi(self,x):
		y1 = (self.e_lam[0]*self.e_mu[0])[None,:]*x[:,None]
		y2 = -.5*self.e_lam[0][None,:]*x[:,None]**2.
		y3 = .5*(self.e_lam[1] - self.e_lam[0]*self.e_mu[1] - np.log(2.*np.pi))
		lnp_xi = (y1 + y2 + y3[None,:])
		return lnp_xi
	
	def calc_lowerbound(self):
		p = self.calc_lnp_xi(self.x)
		self.lowerbound = (p.sum(0) - (self.r*np.log(self.r.sum(0)/self.r.sum())[None,:]).sum(0) - self.lnl_lam - self.lnl_mu).sum()


def run_vmp(x,nstates,init_kmeans=False,maxiters=1000,threshold=1e-16):
	a = vmp_normal(x,nstates)

	if init_kmeans:
		km = kmeans(x,nstates,nrestarts=10)
		xsort = np.argsort(km.mu[:,0])
		mu = km.mu[:,0][xsort]
		var = km.var[:,0,0][xsort] + 1e-300
		
	else:
		mu = np.random.rand(nstates)*(x.max()-x.min()) + x.min()
		mu.sort() # keep states ordered by mu
		# distinv = 1./np.sqrt((x[:,None] - mu[None,:])**2.)
		# r = distinv/distinv.sum(1)[:,None]
		# r = np.sqrt((x[:,None] - mu[None,:])**2.)
		# xr = r.argmin(1)
		# r *= 0.
		# r[xr] = 1.
		#
		# var = np.sum(r*(x[:,None]-mu[None,:])**2.,axis=0)/np.sum(r,axis=0) +1e-300
		var = np.var(x)
	
	a.prior.m = mu
	a.prior.beta  *= .01
	a.prior.b = np.ones(nstates)*var

	lls = np.array((a.lowerbound,))

	for iterations in range(maxiters):
		a.update()
		lls = np.append(lls,a.lowerbound)
		print "%d %.20f"%(iterations,lls[-1])
		
		if (1+iterations) % 5 == 0:
			if np.abs((lls[-5:-1] - lls[-1])/lls[-1]).mean() < threshold:
				break
	return a


#### Testing
import matplotlib.pyplot as plt
from normal_dist import p_normal

n = np.array([200,400])
mu = np.array((5000.,7500.))
x = np.concatenate([np.random.poisson(lam=mu[i],size=n[i]) for i in range(n.size)])

nstates=2
o = kmeans(x,nstates)


# a = vmp_normal(x,nstates)
# a.prior.m = o.mu[:,0]
# a.prior.beta *= 0.01
# a.prior.a = np.ones(nstates)
# a.prior.b = o.var[:,0,0]/100.
#
# for i in range(100):
# 	a.update()
# print a.post.__dict__


hy,hx = plt.hist(x,bins=80,alpha=.5,normed=True)[:2]
xx = np.linspace(0,10000,10000)
# [plt.plot(xx,p_normal(xx,o.mu[i],o.var[i,0,0])*o.pi[i],'k') for i in range(nstates)]
a = [run_vmp(x,2,init_kmeans=True)]
for i in range(4):
	a.append(run_vmp(x,2,init_kmeans=False))
	
cm = plt.cm.spectral
for i in range(len(a)):
	aa = a[i]
	aa.pi = aa.r.sum(0)/aa.r.sum()
	[plt.plot(xx,p_normal(xx,aa.post.m[j],aa.post.b[j]/aa.post.a[j])*aa.pi[j],color=cm(float(i)/float(len(a)-1.))) for j in range(nstates)]
# plt.xlim(hx.min(),hx.max())
plt.show()

print [aa.lowerbound for aa in a]
print [aa.post.m for aa in a]

