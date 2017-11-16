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
		ll = loglikelihood(l,p,z)
		if i > 1 and np.abs((ll-l0)/l0) < 1e-6:
			break
		l0 = ll
	return p

# @nb.jit(nb.double[:,:](nb.int64,nb.double[:,:,:],nb.double,nb.double[:,:]),nopython=True)
# def fit_psf(l,z,s,xys):
# 	## z is (T,X,Y)
# 	## l is int
# 	## s in double
# 	## xy is [x,y]
#
# 	out = np.empty((z.shape[0],xys.shape[1]),dtype=nb.double)
# 	for ii in range(xys.shape[1]):
# 		xy = xys[:,ii]
#
# 		xmin = int(max(0,xy[0]-l))
# 		xmax = int(min(z.shape[0]-1,xy[0]+l) + 1)
# 		ymin = int(max(0,xy[1]-l))
# 		ymax = int(min(z.shape[1]-1,xy[1]+l) + 1)
#
# 		## Find COM
# 		m = np.zeros((3,3))
# 		for t in range(z.shape[0]):
# 			for i in range(3):
# 				for j in range(3):
# 					m[i,j] += z[t,int(xy[0])+i-1,int(xy[1])+j-1]
#
# 		mxy = np.zeros(2)
# 		msum = np.sum(m)
# 		for i in range(3):
# 			for j in range(3):
# 				mm = m[i,j]/msum
# 				x = float(xy[0]+i-1)
# 				y = float(xy[1]+j-1)
# 				mxy[0] += mm*x
# 				mxy[1] += mm*y
#
# 		# Calculate Psi
# 		psi = np.zeros((xmax-xmin,ymax-ymin))
# 		p2sum = 0.
# 		for i in range(xmax-xmin):
# 			for j in range(ymax-ymin):
# 				dex = de(float(xmin + i),mxy[0],s)
# 				dey = de(float(ymin + j),mxy[1],s)
# 				psi[i,j] = dex*dey
# 				p2sum += psi[i,j]**2.
#
# 		## Estimate background and signal
# 		b = np.zeros(z.shape[0],dtype=nb.double)
# 		n = np.zeros_like(b)
# 		n = np.random.rand(z.shape[0])
# 		nxy = (xmax-xmin)*(ymax-ymin)
# 		for t in range(b.size):
# 			b[t] = np.mean(z[t,xmin:xmax,ymin:ymax])
# 			n[t] = z[t,int(mxy[0]),int(mxy[1])]
# 			for it in range(10000):
# 				## update b
# 				for mi in range(xmax-xmin+1):
# 					for mj in range(ymax-ymin+1):
# 						b[t] += 1
# 						b[t] += z[t,int(xmin + mi),int(ymin+mj)] - n[t]*psi[mi,mj]
# 				b[t] /= nxy
# 				## update n
# 				nt = 0.
# 				for mi in range(xmax-xmin+1):
# 					for mj in range(ymax-ymin+1):
# 						nt += (z[t,int(xmin + mi),int(ymin+mj)] - b[t]*psi[mi,mj]) / p2sum
# 				if np.less_equal(abs(nt-n[t]),1e-8 + 1e-5*np.abs(n[t])):
# 					break
# 				else:
# 					n[t] = nt
#
# 		out[:,ii] = n
# 	return out

from scipy.special import erf
from scipy.ndimage import center_of_mass as com
def ml_psf(l,z,sigma,xyi):
	try:
		xmin = int(max(0,xyi[0]-l))
		xmax = int(min(z.shape[0]-1,xyi[0]+l) + 1)
		ymin = int(max(0,xyi[1]-l))
		ymax = int(min(z.shape[1]-1,xyi[1]+l) + 1)

		gx,gy = np.mgrid[xmin:xmax,ymin:ymax]
		gx = gx.astype('f')
		gy = gy.astype('f')
		m = z[:,xmin:xmax,ymin:ymax].astype('f')

		## Find COM
		xyi = com(m.sum(0)) + xyi - l

		dex = .5 * (erf((xyi[0]-gx+.5)/(np.sqrt(2.*sigma**2.))) - erf((xyi[0]-gx -.5)/(np.sqrt(2.*sigma**2.))))
		dey = .5 * (erf((xyi[1]-gy+.5)/(np.sqrt(2.*sigma**2.))) - erf((xyi[1]-gy -.5)/(np.sqrt(2.*sigma**2.))))
		psi = dex*dey

		# b = np.mean(m*(1.-psi[None,:,:]),axis=(1,2))
		b = np.mean(m,axis=(1,2))
		n = z[:,np.round(xyi[0]).astype('i'),np.round(xyi[1]).astype('i')]
		# n = ((m-b[:,None,None])*psi[None,:,:]).sum((1,2))/np.sum(psi**2.)

		n0 = n.sum()
		psum = np.sum(psi**2.)

		for it in xrange(1000):
			b = np.mean(m - n[:,None,None]*psi[None,:,:],axis=(1,2))
			n = np.sum((m - b[:,None,None])*psi[None,:,:],axis=(1,2))/np.sum(psi**2.)
			n1 = n.sum()
			if np.isclose(n1,n0):
				break
			else:
				n0 = n1
	except:
		n = np.zeros(z.shape[0])
		b = np.zeros(z.shape[0])
	return n
