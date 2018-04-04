#### 1D GMM - EM Max Likelihood

import numpy as np
import numba as nb

from fxns.statistics import p_normal
from fxns.kernel_sample import kernel_sample
from fxns.gmm_related import initialize_params, result_ml_gmm

@nb.jit(nb.types.Tuple((nb.float64[:],nb.float64[:],nb.float64[:,:],nb.float64[:],nb.int64,nb.float64))(nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.int64,nb.float64),nopython=True,cache=True)
def outer_loop(x,mu,var,ppi,maxiters,threshold):
	prob = p_normal(x,mu,var)
	r = np.zeros_like(prob)

	p_unif = 1./(x.max()-x.min()) ## Assume data limits define the range
	mu[-1] = 0. ## set outlier class
	var[-1] = 1. ## set outlier class

	iteration = 0
	ll1 = -np.inf
	ll0 = -np.inf

	while iteration < maxiters:
		# E Step
		prob = p_normal(x,mu,var)
		prob[:,-1] = p_unif ## Last state is uniform for outlier detection

		ll0 = ll1
		ll1 = 0
		for i in range(prob.shape[0]):
			ll1i = 0
			for j in range(prob.shape[1]):
				ll1i += ppi[j]*prob[i,j]
			ll1 += np.log(ll1i)

		for i in range(r.shape[0]):
			r[i] = ppi*prob[i]
			r[i] /= np.sum(r[i])

		## likelihood
		if iteration > 1:
			dl = np.abs((ll1 - ll0)/ll0)
			if dl < threshold or np.isnan(ll1):
				break

		## M Step
		nk = np.sum(r,axis=0) + 1e-300
		for i in range(nk.size-1): ## ignore the outlier class
			mu[i] = 0.
			for j in range(r.shape[0]):
				mu[i] += r[j,i]*x[j]
			mu[i] /= nk[i]

			var[i] = 0.
			for j in range(r.shape[0]):
				var[i] += r[j,i]*(x[j] - mu[i])**2.
			var[i] /= nk[i]
		ppi = nk/np.sum(nk)

		iteration += 1
	return mu,var,r,ppi,iteration,ll1



def ml_em_gmm(x,nstates,maxiters=1000,threshold=1e-6):
	'''
	Data convention is NxK

	* Outlier Detection
		Has outlier detection, where the last state is a uniform distribution over the data limits
		mu and var of this state don't mean anything
	'''

	if x.ndim != 1:
		raise Exception("Input data isn't 1D")

	mu,var,ppi = initialize_params(x,nstates)
	mu,var,r,ppi,iteration,ll1 = outer_loop(x,mu,var,ppi,maxiters,threshold)
	result = result_ml_gmm(mu,var,r,ppi,ll1,iteration)

	return result



# ### Examples
# from fxns.fake_data import fake_data
# t,d = fake_data(outliers=True)
#
# nstates = 5
# o = ml_em_gmm(d,nstates)
#
# import matplotlib.pyplot as plt
# cm = plt.cm.jet
# for i in range(nstates):
# 	c = cm(float(i)/nstates)
# 	# print i,c
# 	xcut = o.r.argmax(1) == i
# 	plt.plot(t[xcut],d[xcut],'o',color=c,alpha=.5)
# plt.show()
