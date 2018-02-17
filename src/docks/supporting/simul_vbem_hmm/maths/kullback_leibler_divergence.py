import numpy as np
from math_fxns import psi,gammaln
import numba as nb

@nb.jit(nb.double(nb.double,nb.double),nopython=True)
def wishart_ln_B(a,b):
	return -a*np.log(b) - gammaln(a)

@nb.jit(nb.double(nb.double,nb.double),nopython=True)
def wishart_entropy(a,b):
	return - wishart_ln_B(a,b) - (a-1.)*(psi(a)-np.log(b)) + a

@nb.jit(nb.double(nb.double[:],nb.double[:],nb.double[:],nb.double[:],nb.double[:],nb.double[:],nb.double[:],nb.double[:]),nopython=True)
def dkl_normalgamma(mn,betan,an,bn,m0,beta0,a0,b0):
	k = mn.size

	# Bishop 10.74
	E_lnp_mulam = 0.
	for i in range(k):
		E_lnp_mulam += 0.5 * np.log(beta0[i]/(2.*np.pi))
		E_lnp_mulam += 0.5 * (psi(an[i]) - np.log(bn[i]))
		E_lnp_mulam -= 0.5 * beta0[i]*(1./betan[i] + an[i]/bn[i]*(mn[i]-m0[i])**2.)
		E_lnp_mulam += wishart_ln_B(a0[i],b0[i])
		E_lnp_mulam += (a0[i]-1.)*(psi(an[i]) - np.log(bn[i]))
		E_lnp_mulam -= an[i]*b0[i]/bn[i]

	# Bishop 10.77
	E_lnq_mu_lam = 0.
	for i in range(k):
		E_lnq_mu_lam += 0.5 * (psi(an[i]) - np.log(bn[i]))
		E_lnq_mu_lam += 0.5 * (np.log(betan[i]/(2.*np.pi)) - 1.)
		E_lnq_mu_lam -= wishart_entropy(an[i],bn[i])

	return E_lnp_mulam - E_lnq_mu_lam

@nb.jit(nb.double(nb.double[:],nb.double[:]),nopython=True)
def dkl_dirichlet(un,u0):
	s_un = np.sum(un)
	s_u0 = np.sum(u0)

	y  =  gammaln(s_un)  - np.sum(gammaln(un))
	y += -gammaln(s_u0) + np.sum(gammaln(u0))
	y += np.sum((un-u0)*(psi(un) - psi(s_un)))
	return y

@nb.jit(nb.double(nb.double[:,:],nb.double[:,:]),nopython=True)
def dkl_tmatrix(alphan,alpha0):
	k,_ = alphan.shape
	y = 0.
	for i in range(k):
		y += dkl_dirichlet(alphan[i],alpha0[i])
	return y
