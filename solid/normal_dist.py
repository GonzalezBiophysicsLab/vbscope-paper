import numpy as np
from scipy import special
from scipy.optimize import minimize


def ln_p_normal(x,mu,var):
	y = np.log(2.*np.pi) + np.log(var) + (x-mu)**2./var
	return -.5 * y * (var > 0.)

def p_normal(x,mu,var):
	return np.exp(ln_p_normal(x,mu,var))

###############

def ln_maxval_normal(x,n,mu,var):
	prec = 1./var
	y = .5+.5*special.erf((x-mu)*np.sqrt(prec/2.))
	return (n-1.)*np.log(y) + np.log(n) + ln_p_normal(x,mu,var)
	# return n*(y**(n-1.))*p_normal(x,mu,var)

def ln_minval_normal(x,n,mu,var):
	prec = 1./var
	y = .5+.5*special.erf(-(x-mu)*np.sqrt(prec/2.))
	return (n-1.)*np.log(y) + np.log(n) + ln_p_normal(x,mu,var)
	# return n*(y**(n-1.))*p_normal(x,mu,var)
	
def maxval_normal(x,n,mu,var):
	return np.exp(ln_maxval_normal(x,n,mu,var))
def minval_normal(x,n,mu,var):
	return np.exp(ln_minval_normal(x,n,mu,var))

###############


def fit_maxval_normal(d,n,x0=None):
	def _fxn(muvar):
		if muvar[1] <= 0.:
			return np.inf
		y = -np.log(maxval_normal(d, n, muvar[0], muvar[1]))
		return y.sum()
		
	if x0 is None:
		x0 = [d.min(), np.var(d)]
	out = minimize(_fxn, x0=x0, method='Nelder-Mead',options={'maxfev':1000})#,callback=_fit_callback)

	if out.success:
		return out.x
	return np.array([np.nan,np.nan])

def fit_minval_normal(d,n,x0=None):
	def _fxn(muvar):
		if muvar[1] <= 0.:
			return np.inf
		y = -np.log(minval_normal(d, n, muvar[0], muvar[1]))
		return y.sum()
		
	if x0 is None:
		x0 = [d.max(), np.var(d)]
	out = minimize(_fxn, x0=x0, method='Nelder-Mead',options={'maxfev':1000})#,callback=_fit_callback)

	if out.success:
		return out.x
	return np.array([np.nan,np.nan])