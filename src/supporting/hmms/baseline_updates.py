import numpy as np
import numba as nb

@nb.njit
def tridiagonal_solver(a,b,c,d):
	## Non destructive

	cp = np.zeros_like(c)
	dp = np.zeros_like(d)
	x = np.zeros_like(d)

	cp[0] = c[0]/b[0]
	dp[0] = d[0]/b[0]
	for i in range(1,d.size):
		cp[i] = c[i]/(b[i]-a[i]*cp[i-1])
		dp[i] = (d[i]-a[i]*dp[i-1])/(b[i]-a[i]*cp[i-1])

	x[-1] = dp[-1]
	for j in range(x.size-1):
		i = x.size - 2 -j
		x[i] = dp[i] - cp[i]*x[i+1]

	return x

@nb.njit(nb.double[:](nb.double[:],nb.double[:],nb.double))
def update_baseline(data,model,r2):
	rhs = -r2*(data-model)
	maind = np.zeros_like(data) -r2 - 2.
	maind[0] = -r2 - 1.
	maind[-1] = -r2 - 1.
	lower = np.ones_like(data)
	upper = np.ones_like(data)

	return tridiagonal_solver(lower,maind,upper,rhs)

@nb.njit(nb.double(nb.double[:],nb.double[:],nb.double[:],nb.double))
def update_vn(data,model,background,r2):
	vn = np.mean((data - background - model)**2.) + np.mean((background[1:]-background[:-1])**2.)/r2
	return vn

@nb.njit(nb.double(nb.double[:],nb.double[:],nb.double[:]))
def update_r2(data,model,background):
	B = np.sum((data - background - model)**2.) / np.sum((background[1:]-background[:-1])**2.)
	left = 0.
	right = 1e5
	thresh = 1e-8

	while 1:
		r2 = (right+left)/2.
		val = r2**3. - B * (4. + r2)
		if np.abs(val) < thresh:
			break

		if val > 0:
			right = r2
		else:
			left = r2
	return r2

@nb.njit(nb.double(nb.double[:],nb.double[:],nb.double[:],nb.double,nb.double))
def calc_ll(data,model,baseline,vn,r2):
	vb = vn*r2
	n = data.size

	out = -.5*np.sum((baseline[1:]-baseline[:-1])**2.)/vb -.5*np.sum((data-baseline-model)**2.)/vn
	out += -.5*n*np.log(2.*np.pi*vn)
	out += -n*np.log(.5*np.sqrt(r2)+np.sqrt(r2+4.))

	return out/n
