#### 1D HMM - EM Max Likelihood

import numpy as np
import numba as nb
from sys import platform
import multiprocessing as mp

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

	# from ml_em_gmm import ml_em_gmm
	# o = ml_em_gmm(x,nstates+1)
	# mu = o.mu[:-1]
	# var = o.var[:-1]
	# ppi = o.ppi[:-1]
	# ppi /= ppi.sum() ## ignore outliers

	mu,var,ppi = initialize_params(x,nstates)
	tmatrix = initialize_tmatrix(nstates)

	mu,var,r,ppi,tmatrix,iteration,likelihood = outer_loop(x,mu,var,ppi,tmatrix,maxiters,threshold)

	result = result_ml_hmm(mu,var,r,ppi,tmatrix,np.array((likelihood)),iteration)
	result.viterbi = viterbi(x,result.mu,result.var,result.tmatrix,result.ppi)

	return result

def ml_em_hmm_parallel(x,nstates,maxiters=1000,threshold=1e-10,nrestarts=1,ncpu=1):

	if platform != 'win32' and ncpu != 1 and nrestarts != 1:
		pool = mp.Pool(processes = ncpu)
		results = [pool.apply_async(ml_em_hmm, args=(x,nstates,maxiters,threshold)) for i in xrange(nrestarts)]
		results = [p.get() for p in results]
		pool.close()
	else:
		results = [ml_em_hmm(x,nstates,maxiters,threshold) for i in xrange(nrestarts)]

	try:
		best = np.nanargmax([r.likelihood for r in results])
	except:
		best = 0
	return results[best]
