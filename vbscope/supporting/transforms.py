import numpy as np
from skimage.transform import AffineTransform,EuclideanTransform,PolynomialTransform


def icp(reference,transformee,xshift=0.,yshift=0.,maxiters=100):
	'''
	both inputs are Nx2 vectors w/ coordinates of points. can be different numbers of points

	* reference is the set which everything will be transformed into
	* transformee is the set that will be transformed

	returns: the scikit-image affine transform function. To use this, input
	coordinates to be transformed (i.e., transformee) in Nx2 form... and it will return the transformed coordinates
	'''

	c1 = reference.T
	c2 = transformee.T

	if xshift == 0. and yshift == 0.:
		t = EuclideanTransform(translation=np.median(c1,axis=1) - np.median(c2,axis=1))
	else:
		t = EuclideanTransform(translation=[xshift,yshift])

	last = np.inf
	for i in range(maxiters):
		ct2 = t(c2.T).T
		dr = np.sqrt((ct2[0,None,:] - c1[0,:,None])**2. + (ct2[1,None,:] - c1[1,:,None])**2.)
		a = np.argmax(dr.shape)
		rmin = dr.argmin(a)
		r = dr.min(a)
		if a == 0:
			dst = c2
			src = c1[:,rmin]
		else:
			dst = c2[:,rmin]
			src = c1
		t.estimate(dst.T,src.T)
		l = np.median(t.residuals(dst.T,src.T))
		# print i,l
		if np.isclose(l,last):
			return t
		else:
			last = l
	print("Didn't converge... which is weird.")
	return t

def poly(src,dst,order=2):
	t =PolynomialTransform()
	t.estimate(dst,src,order=order)
	return t

def affine(src,dst):
	t = AffineTransform()
	t.estimate(dst,src)
	return t

def get_closest(c1,c2,transform):
	ct2 = transform(c2.T).T
	dr = np.sqrt((ct2[0,None,:] - c1[0,:,None])**2. + (ct2[1,None,:] - c1[1,:,None])**2.)
	a = np.argmax(dr.shape)
	rmin = dr.argmin(a)
	# r = np.arange(np.min(dr.shape))
	return a,rmin

def interpolated_fft_phase_alignment(d1,d2):
	from skimage import transform
	from skimage.transform import SimilarityTransform,warp
	from scipy.interpolate import RectBivariateSpline

	## Calculate Cross-correlation
	f1 = np.fft.fft2(d1-d1.mean())
	f2 = np.fft.fft2(d2-d2.mean())
	f = np.conjugate(f1)*f2 # / (np.abs(f1)*np.abs(f2))
	d = np.fft.ifft2(f).real

	## Find maximum in c.c.
	s1,s2 = np.nonzero(d==d.max())

	#### Interpolate for finer resolution
	## cut out region
	l = 5.
	xmin = int(np.max((0.,s1-l)))
	xmax = int(np.min((d.shape[0],s1+l)))
	ymin = int(np.max((0.,s2-l)))
	ymax = int(np.min((d.shape[1],s2+l)))
	dd = d[xmin:xmax,ymin:ymax]

	## calculate interpolation
	x = np.arange(xmin,xmax)
	y = np.arange(ymin,ymax)
	interp = RectBivariateSpline(x,y,dd)

	## interpolate new grid
	x = np.linspace(xmin,xmax,(xmax-xmin)*10)
	y = np.linspace(ymin,ymax,(ymax-ymin)*10)
	di = interp(x,y)
	# return di

	## get maximum in interpolated c.c. for sub-pixel resolution
	xy = np.nonzero(di==di.max())

	## get interpolated shifts
	s1 = x[xy[0][0]]
	s2 = y[xy[1][0]]
	if s1 > d.shape[0]/2:
		s1 -= d.shape[0]
	if s2 > d.shape[1]/2:
		s2 -= d.shape[1]

	## calculate warped img
	tform = SimilarityTransform(translation=(-s2,-s1))
	img = warp(d2, tform)

	return -s1,-s2,img,tform


### Example:
# from icp import icp
#
# transform = icp(c1.T,c2.T)
#
# plt.figure()
# plt.plot(c1[0],c1[1],'o')
# plt.plot(c2[0],c2[1],'o')
# ct2 = transform(c2.T).T
# plt.plot(ct2[0],ct2[1],'o')
# plt.show()
#
