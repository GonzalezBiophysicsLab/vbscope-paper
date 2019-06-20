import numpy as np
from scipy.interpolate import RectBivariateSpline

def _get_shift(di,f0,hdx,hdy):
	## Calculate crosscorrelation
	f2 = np.fft.fft2(di-di.mean())
	f = f0*f2
	dd = np.fft.fftshift(np.fft.ifft2(f).real)

	## Find maximum in c.c.
	s1,s2 = np.nonzero(dd==dd.max())

	#### Interpolate for finer resolution
	## cut out region
	l = 5.
	xmin = int(np.max((0.,s1-l)))
	xmax = int(np.min((dd.shape[0],s1+l)))
	ymin = int(np.max((0.,s2-l)))
	ymax = int(np.min((dd.shape[1],s2+l)))
	ddd = dd[xmin:xmax,ymin:ymax]

	## calculate interpolation
	x = np.arange(xmin,xmax)
	y = np.arange(ymin,ymax)
	interp = RectBivariateSpline(x,y,ddd)

	## interpolate new grid
	x = np.linspace(xmin,xmax,(xmax-xmin)*100+1)
	y = np.linspace(ymin,ymax,(ymax-ymin)*100+1)
	di = interp(x,y)

	## get maximum in interpolated c.c. for sub-pixel resolution
	xy = np.nonzero(di==di.max())
	## get interpolated shifts
	s1 = x[xy[0][0]] - hdx
	s2 = y[xy[1][0]] - hdy
	if s1 > dd.shape[0]/2:
		s1 -= dd.shape[0]
	if s2 > dd.shape[1]/2:
		s2 -= dd.shape[1]

	return np.array((s1,s2))

def _translate_frame(d,shift):
	from skimage.transform import SimilarityTransform,warp
	tform = SimilarityTransform(translation=(shift[0],shift[1]))
	return warp(d.astype('float64'), tform,mode='symmetric').astype('d')

def align_stack(d,reference=0,clip=10,filter_width=5,callback=lambda : True):
	'''
	Calculate the XY translations (shifts) of each image relative to the reference frame

	d - data - (N,X,Y)
	reference - reference frame number - int
	clip - number of border pixels in each direction - int
	filter_width - number pixels to calculate moving median filter the resulting shifts. This removes outliers
	'''

	## Precompute
	f0 = np.fft.fft2(d[reference,clip:-clip,clip:-clip]-d[reference,clip:-clip,clip:-clip].mean()).conj()
	hdx = f0.shape[0]//2
	hdy = f0.shape[1]//2

	shifts = np.zeros((d.shape[0],2))
	for i in range(d.shape[0]):
		shifts[i] = _get_shift(d[i,clip:-clip,clip:-clip],f0,hdx,hdy)
		if not callback():
			break
	from scipy import ndimage as nd
	shifts = nd.median_filter(shifts,(3,1))
	shifts = nd.gaussian_filter(shifts,(filter_width,1))
	return shifts

# def align_stack(d,reference=0,clip=10,filter_width=5,callback=lambda : True):
# 	'''
# 	Calculate the XY translations (shifts) of each image relative to the reference frame
#
# 	d - data - (N,X,Y)
# 	reference - reference frame number - int
# 	clip - number of border pixels in each direction - int
# 	filter_width - number pixels to calculate moving median filter the resulting shifts. This removes outliers
# 	'''
#
# 	## Precompute
# 	#### sum approach inspired by motioncorr2
# 	dsum = d.sum(0)[clip:-clip,clip:-clip].astype('float64')
# 	hdx = dsum.shape[0]//2
# 	hdy = dsum.shape[1]//2
#
# 	shifts = np.zeros((d.shape[0],2))
# 	for i in range(d.shape[0]):
# 		dd = dsum - d[reference,clip:-clip,clip:-clip]
# 		dd -= dd.mean()
# 		f0 = np.fft.fft2(dd).conj()
#
# 		shifts[i] = _get_shift(d[i,clip:-clip,clip:-clip].astype('float64'),f0,hdx,hdy)
# 		if not callback():
# 			break
# 	from scipy import ndimage as nd
# 	# shifts = nd.median_filter(shifts,(filter_width,1))
# 	shifts = nd.gaussian_filter(shifts,(filter_width,1))
# 	return shifts

def dedrift(d,shifts,callback=lambda : True):
	'''
	Warp every frame in d according to translations in shifts

	d - data - (N,X,Y)
	shifts - xy translations - (N,2)
	'''

	for i in range(d.shape[0]):
		d[i] = _translate_frame(d[i],shifts[i])
		if not callback():
			break
	return d
