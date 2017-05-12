import numpy as np
from scipy import special
from scipy.optimize import minimize


def ln_p_poisson(k,lam):
	y = k*np.log(lam) - lam - special.gammaln(k+1.)
	return y + np.log(k>=0.) + np.log(lam > 0.)

def p_poisson(k,lam):
	return np.exp(ln_p_poisson(k,lam))

###############

def ln_maxval_poisson(k,n,lam):
	y = (n-1.)*np.log(special.gammaincc(k+1.,lam)) + np.log(n) + ln_p_poisson(k,lam)
	return y

def ln_minval_poisson(k,n,lam):
	y = (n-1.)*np.log(special.gammainc(k+1.,lam)) + np.log(n) + ln_p_poisson(k,lam)
	return y

###############

def fit_maxval_poisson(d,n,x0=None):
	def _fxn(lambg):
		y = -ln_maxval_poisson(d+lambg[1], n, lambg[0])
		return y.sum()
		
	if x0 is None:
		x0 = np.array([d.min(),np.var(d)/np.sqrt(n)-d.mean()])
	out = minimize(_fxn, x0=x0, method='Nelder-Mead',options={'maxfev':1000})#,callback=_fit_callback)

	if out.success:
		return out.x
	return np.array([np.nan,np.nan])

def fit_minval_poisson(d,n,x0=None):
	def _fxn(lambg):
		y = -ln_minval_poisson(d+lambg[1], n, lambg[0])
		return y.sum()
		
	if x0 is None:
		x0 = np.array([d.max(),np.var(d)/np.sqrt(n)-d.mean()])
	out = minimize(_fxn, x0=x0, method='Nelder-Mead',options={'maxfev':1000})#,callback=_fit_callback)

	if out.success:
		return out.x
	return np.array([np.nan,np.nan])