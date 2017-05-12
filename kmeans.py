import numpy as np

class _results_kmeans(object):
	def __init__(self,nstates,pi,r,mu,var):
		self.nstates = nstates
		self.pi = pi
		self.r = r
		self.mu = mu
		self.var = var
	
	def sort(self):
		xsort = self.pi.argsort()[::-1]
		# xsort = np.argsort(self.mu)
		self.pi = self.pi[xsort]
		self.r = self.r[:,xsort]
		self.var = self.var[xsort]
		self.mu = self.mu[xsort]
		

def kmeans(x,nstates,nrestarts=1):
	"""
	Multidimensional, K-means Clustering
	
	Input:
		* `x` is an (N,d) `np.array`, where N is the number of data points and d is the dimensionality of the data
		* `nstates` is the K in K-means
		* `nrestarts` is the number of times to restart. The minimal variance results are provided

	Returns:
		* a `_results_kmeans` object that contains
			- `pi` - k - the probability of each state
			- `r` - Nk -  the responsibilities of each data point
			- `mu` - kd - the means
			- `var_k` - kdd - the covariances
	"""

	if x.ndim == 1:
		x = x[:,None]

	jbest = np.inf
	mbest = None
	rbest = None
	for nr in range(nrestarts):
		mu_k = x[np.random.randint(0,x.shape[0],size=nstates)]
		j_last = np.inf
		for i in range(500):
			dist = np.sqrt(np.sum(np.square(x[:,None,:] - mu_k[None,...]),axis=2))
			r_nk = (dist == dist.min(1)[:,None]).astype('i')
			j = (r_nk.astype('f') * dist).sum()
			mu_k = (r_nk[:,:,None].astype('f')*x[:,None,:]).sum(0)/(r_nk.astype('f').sum(0)[:,None]+1e-16)
			if np.abs(j - j_last)/j <= 1e-100:
				if j < jbest:
					jbest = j
					mbest = mu_k
					rbest = r_nk
				break
			else:
				j_last = j
	mu_k = mbest
	r_nk = rbest
	sig_k = np.empty((nstates,x.shape[1],x.shape[1]))
	for k in range(nstates):
		sig_k[k] = np.cov(x[r_nk[:,k]==1].T)
	pi_k = (r_nk.sum(0)).astype('f')
	pi_k /= pi_k.sum()

	#pi_k is fraction, r_nk is responsibilities, mu_k is means, sig_k is variances
	results = _results_kmeans(nstates,pi_k,r_nk,mu_k,sig_k)
	results.sort()
	return results