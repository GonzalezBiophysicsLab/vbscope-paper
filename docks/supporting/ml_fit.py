import numpy as np
from math import erf
import numba as nb

@nb.jit("double(double,double,double)")
def de(x,m,s):
	return .5 * (erf((x-m+.5)/(np.sqrt(2.*s*s)))
	- erf((x-m -.5)/(np.sqrt(2.*s*s))))

@nb.jit("double(double,double,double,double,double)",nopython=True)
def g_rnm(r,n,m,rm,rs):
	g = 1. / np.sqrt(2. * np.pi) / (rs**n)
	g *= ((r - rm - .5)**m * np.exp(-.5 / (rs**2.) * (r-rm-.5)**2.)
		- (r - rm + .5)**m * np.exp(-.5 / (rs**2.) * (r-rm+.5)**2.))
	return g

@nb.jit(nb.double(nb.int64,nb.double[:],nb.double[:,:]),nopython=True)
def loglikelihood(l,p,z):
	mx,my,sx,sy,i0,b,x0,y0 = p

	xmin = int(max(0,x0-l))
	xmax = int(min(z.shape[0]-1,x0+l) + 1)
	ymin = int(max(0,y0-l))
	ymax = int(min(z.shape[1]-1,y0+l) + 1)

	ll = 0.
	for i in range(xmax-xmin):
		for j in range(ymax-ymin):
			# Calcs
			x = xmin + i
			y = ymin + j
			zz = z[x,y]
			dex = de(float(x),mx,sx)
			dey = de(float(y),my,sy)
			m = i0 * dex * dey + b

			ll += zz*np.log(m) - m - (zz*np.log(zz) - zz)
	return ll

@nb.jit(nb.double[:](nb.int64,nb.double[:],nb.double[:,:]),nopython=True)
def update(l,p,z):
	mx,my,sx,sy,i0,b,x0,y0 = p
	pp = np.array((mx,my,i0,b))

	xmin = int(max(0,x0-l))
	xmax = int(min(z.shape[0]-1,x0+l) + 1)
	ymin = int(max(0,y0-l))
	ymax = int(min(z.shape[1]-1,y0+l) + 1)

	f1 = np.zeros(pp.size)
	f2 = np.zeros(pp.size)
	for i in range(xmax-xmin):
		for j in range(ymax-ymin):
			# Calcs
			x = float(xmin + i)
			y = float(ymin + j)
			dex = de(x,mx,sx)
			dey = de(y,my,sy)
			m = i0 * dex * dey + b
			zz = z[xmin+i,ymin+j]

			# Calc first derivs
			dmx = i0 * dey * g_rnm(x,1.,0.,mx,sx)
			dmy = i0 * dex * g_rnm(y,1.,0.,my,sy)
			di0 = dex*dey
			db = 1.
			d = np.array((dmx,dmy,di0,db))

			# Calc second derivs
			ddmx = i0 * dey * g_rnm(x,3.,1.,mx,sx)
			ddmy = i0 * dex * g_rnm(y,3.,1.,my,sy)
			ddi0 = 0.
			ddb  = 0.
			dd = np.array((ddmx,ddmy,ddi0,ddb))

			# Updates
			for k in range(f1.size):
				f1[k] += d[k] * (zz/m-1.)
				f2[k] += dd[k] * (zz/m-1.) - d[k]**2. * zz/m/m

			jumpmax = 0.3
			for k in range(f1.size):
				jump = f1[k]/(f2[k]+1e-300)
				if np.abs(jump) > jumpmax*pp[k]:
					jump = np.sign(jump)*jumpmax*pp[k]
				pp[k] -= jump

	p[0] = pp[0] # mx
	p[1] = pp[1] # my
	p[4] = pp[2] # i0
	p[5] = pp[3] # b

	return p

@nb.jit(nb.double[:](nb.int64,nb.double[:,:],nb.double,nb.double[:]),nopython=True)
def fit(l,z,s,xy):
	gb = np.min(z[int(xy[0])-1:int(xy[0])+2,int(xy[1])-1:int(xy[1])+2])
	gi = z[int(xy[0]),int(xy[1])] - gb
	p = np.array((xy[0],xy[1],s,s,gi,gb,xy[0],xy[1]))

	l0 = -np.inf
	ll = loglikelihood(l,p,z)
	for i in range(20):
		p = update(l,p,z)
		# ll = loglikelihood(l,p,z)
		# if i > 1 and np.abs((ll-l0)/l0) < 1e-6:
		# 	break
		# l0 = ll
	return p
