import numpy as np
from scipy import special

##### Distributions
## Extreme value distribution PDFs go as
## p(x) = (n-1)*F(x)^(n-1)*f(x),
## where F(x) is the CDF,
## and f(x) is the PDF of the regular distribution (i.e., Normal here...)

def lnp_normal(x,mu,var):
	y = np.log(2.*np.pi) + np.log(var) + (x-mu)**2./var
	return -.5 * y * (var > 0.)

def p_normal(x,mu,var):
	return np.exp(lnp_normal(x,mu,var))

def lnp_normal_min(x,n,mu,var):
	prec = 1./var
	y = .5+.5*special.erf(-(x-mu)*np.sqrt(prec/2.))
	return (n-1.)*np.log(y) + np.log(n) + lnp_normal(x,mu,var)

def lnp_normal_max(x,n,mu,var):
	prec = 1./var
	y = .5+.5*special.erf((x-mu)*np.sqrt(prec/2.))
	return (n-1.)*np.log(y) + np.log(n) + lnp_normal(x,mu,var)

def p_normal_max(x,n,mu,var):
	return np.exp(lnp_normal_max(x,n,mu,var))

def p_normal_min(x,n,mu,var):
	return np.exp(lnp_normal_min(x,n,mu,var))

##### Moments

def normal_order_stat(r,n,mu,var,alpha=.375):
	#### Approximate formula is from:
	# Algorithm AS 177: Expected Normal Order Statistics (Exact and Approximate)
	# Author(s): J. P. Royston
	# Source: Journal of the Royal Statistical Society. Series C (Applied Statistics), Vol. 31, No. 2
	# (1982), pp. 161-165
	# Stable URL: http://www.jstor.org/stable/2347982

	return mu + special.ndtri((r-alpha)/(n-2.*alpha+1.))*np.sqrt(var)

def _get_alpha(n):
	### Numerical minimization of Monte Carlo samples gave me:
	##  n | n^2 |  \alpha
	## ------------------
	##  3     9   0.36205
	##  5    25   0.37662
	##  7    49   0.38432
	##  9    81   0.38897
	## 11   121   0.39264

	if n == 9:
		alpha = .36205
	elif n == 25:
		alpha = 0.37662
	elif n == 49:
		alpha = .38432
	elif n == 81:
		alpha = .38897
	elif n == 121:
		alpha = .39264
	else:
		alpha = .375
	return alpha

#### Estimations

def _estimate_mu(n,var):
	alpha = _get_alpha(n)
	return special.ndtri((n-alpha)/(n-2.*alpha+1.))*np.sqrt(var)

def _estimate_var(n):
	### Parameters from numerical optimization of Monte Carlo Samples
	# From guessing the form: (a+b/n)/(n*ln(n)), which seems to linearize well

	a = .85317
	b = -.573889
	y = np.log(n)/(a+b/n)
	return y

def estimate_from_min(d,n):
	v = np.var(d)  * _estimate_var(n)
	m = np.mean(d) + _estimate_mu(n,v)
	return m,v

def estimate_from_max(d,n):
	v = np.var(d)  * _estimate_var(n)
	m = np.mean(d) - _estimate_mu(n,v)
	return m,v
