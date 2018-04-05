#### 1D HMM - EM Max Likelihood

import numpy as np
import numba as nb

from fxns.numba_math import psi
from fxns.statistics import p_normal,dirichlet_estep
from fxns.hmm_related import forward_backward,viterbi,result_ml_hmm,initialize_tmatrix
from fxns.kernel_sample import kernel_sample
from fxns.gmm_related import initialize_params



@nb.jit(nb.types.Tuple((nb.float64[:],nb.float64[:],nb.float64[:,:],nb.float64[:],nb.float64[:,:],nb.int64,nb.float64))(nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:,:],nb.int64,nb.float64),nopython=True)
def outer_loop(x,mu,var,ppi,tmatrix,maxiters,threshold):
	prob = p_normal(x,mu,var)

	iteration = 0
	ll1 = -np.inf
	ll0 = -np.inf
	while iteration < maxiters:
		## E Step
		# Forward-Backward
		ll0 = ll1
		prob = p_normal(x,mu,var)
		r, xi, ll1 = forward_backward(prob, tmatrix, ppi)

		## likelihood
		if iteration > 1:
			dl = np.abs((ll1 - ll0)/ll0)
			if dl < threshold:
				break

		## M Step
		nk = np.sum(r,axis=0) + 1e-300
		for i in range(nk.size):
			mu[i] = np.sum(r[:,i]*x)/nk[i]
			var[i] = np.sum(r[:,i]*(x - mu[i])**2.)/nk[i]
		ppi = nk/nk.sum()

		for i in range(tmatrix.shape[0]):
			for j in range(tmatrix.shape[1]):
				tmatrix[i,j] = np.mean(xi[:,i,j])
		for i in range(tmatrix.shape[0]):
			tmatrix[i] /= np.sum(tmatrix[i])

		iteration += 1
	return mu,var,r,ppi,tmatrix,iteration,ll1


def ml_em_hmm(x,nstates,maxiters=1000,threshold=1e-6):
	'''
	Convention is NxK
	'''

	if x.ndim != 1:
		raise Exception("Input data isn't 1D")

	from ml_em_gmm import ml_em_gmm
	o = ml_em_gmm(x,nstates)
	mu = o.mu
	var = o.var
	ppi = o.ppi
	tmatrix = initialize_tmatrix(nstates)

	mu,var,r,ppi,tmatrix,iteration,likelihood = outer_loop(x,mu,var,ppi,tmatrix,maxiters,threshold)

	result = result_ml_hmm(mu,var,r,ppi,tmatrix,likelihood,iteration)
	result.viterbi = viterbi(x,result.mu,result.var,result.tmatrix,result.ppi)

	return result
