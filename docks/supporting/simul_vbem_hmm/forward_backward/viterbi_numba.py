#viterbi numba
import numba as nb
import numpy as np

# @nb.jit(nb.int64[:](nb.double[:,:],nb.double[:,:],nb.double[:]),nopython=True)
# def viterbi(p_x_z,A,pi):
# 	n,k = p_x_z.shape
#
# 	omega = np.zeros((n,k))
# 	zmax = np.zeros((n,k),dtype=nb.int64)
# 	zhat = np.zeros(n,dtype=nb.int64)
#
# 	for i in range(k):
# 		omega[0,i] = pi[i] + p_x_z[0,i]
#
# 	for t in range(1,omega.shape[0]):
# 		for i in range(k):
# 			q = A[0,i] + omega[t-1][0]
# 			for j in range(k):
# 				if A[j,i] + omega[t-1][j] > q:
# 					q = A[j,i] + omega[t-1][j]
# 			q = np.max(A[:,i]+omega[t-1])
# 			omega[t,i] = q + p_x_z[t,i]
#
# 			for j in range(k):
# 				if A[j,i] + omega[t-1,j] > A[zmax[t,i],i] + omega[t-1,zmax[t,i]]:
# 					zmax[t,i] = j
#
# 	for i in range(k):
# 		if omega[n-1,zhat[n-1]] < omega[n-1,i]:
# 			zhat[n-1] = i
# 	for tt in range(n-1):
# 		t = n-2-tt
# 		zhat[t] = zmax[t+1,zhat[t+1]]
#
# 	return zhat
#
# def viterbi(p_x_z,A,pi):
# 	D = 1
# 	T,K = p_x_z.shape
# 	omega = np.zeros((T,K))
# 	bestPriorZ = np.zeros((T,K))
# 	z_hat = np.zeros((1,T),dtype='i')
#
# 	pZ0 = pi / pi.sum()
# 	A /= A.sum(1)[:,None]
#
#
# 	for k in range(K):
# 		omega[0,k] = np.log(pZ0[k]) + np.log(p_x_z[0,k])
#
# 	bestPriorZ[0,:] = 0.
#
# 	for t in range(T-1):
# 		for k in range(K):
# 			omega[t+1,k] = np.log(p_x_z[t+1,k]) + np.max((np.log(A)))
# 	% Compute values for timesteps 2-end.
# 	% omega(zn)=ln(p(xn|zn))+max{ln(p(zn|zn-1))+omega(zn-)}
# 	% CB 13.68
# 	for t=2:T
# 		for k=1:K
#
# 	        [omega(t,k) bestPriorZ(t,k)] =max(log(A(:,k)')+omega(t-1,:));
# 	        omega(t,k) = omega(t,k)+ log(gauss(mus(:,k), covarMtx(:,:,k),x(:,t)'));
# 	    end
# 	end
#
# 	[logLikelihood z_hat(T)]=max(omega(T,:));
# 	for t=(T-1):-1:1
# 	    z_hat(t) = bestPriorZ(t+1,z_hat(t+1));
# 	end
# 	x_hat=mus(:,z_hat)';

def viterbi(pxz,alpha,rho):
	### v is vbhmm_result

	ln_p_x_z = np.log(pxz)
	ln_A = np.log(alpha/alpha.sum(1)[:,None])
	ln_ppi = np.log(rho/rho.sum())

	omega = np.empty_like(ln_p_x_z)
	zmax = np.empty_like(ln_p_x_z,dtype='i')
	zhat = np.empty(ln_p_x_z.shape[0],dtype='i')

	omega[0] = ln_ppi + ln_p_x_z[0]
	for t in range(1,omega.shape[0]):
		omega[t] = ln_p_x_z[t] + np.max(ln_A + omega[t-1][:,None],axis=0)
		zmax[t] = np.argmax(ln_A + omega[t-1][:,None],axis=0)

	zhat[-1] = omega[-1].argmax()
	for t in range(zhat.shape[0])[::-1][1:]:
		zhat[t] = zmax[t+1,zhat[t+1]]

	return zhat
