import numpy as np
import numba as nb

@nb.njit
def make_shift(image,s):
	nx,ny = image.shape
	ds = (s-1)//2
	shifted = np.zeros((s*s,nx,ny))

	for i in range(nx):
		for j in range(ny):
			for ii in range(-ds,ds+1):
				for jj in range(-ds,ds+1):
					if ii + i >= 0 and ii + i < nx and jj + j >= 0 and jj + j < ny:
						shifted[(ds+ii)*s+(ds+jj),i,j] = image[ii+i,jj+j]
	return shifted

@nb.njit
def clip(image,s):
	iimage = np.copy(image)
	iimage[:s] = 0
	iimage[-s:] = 0
	iimage[:,:s] = 0
	iimage[:,-s:] = 0
	return iimage

def max_map(image,s):
	shifted = make_shift(image,s)
	middle = (s+1)*(s-1)/2
	return clip(np.argmax(shifted,axis=0) == middle,s)

def min_map(image,s):
	shifted = make_shift(image,s)
	middle = (s+1)*(s-1)/2
	return clip(np.argmin(shifted,axis=0) == middle,s)

# @nb.njit
def minmax_map(image,s,clip_level=None):
	if clip_level is None:
		clip_level = s

	mmax = max_map(image,s)
	mmin = min_map(image,s)
	# shifted = make_shift(image,s)
	#
	# mmax = np.zeros_like(image,dtype='bool')
	# mmin = np.zeros_like(image,dtype='bool')
	# for i in range(image.shape[0]):
	# 	for j in range(image.shape[1]):
	# 		if np.argmax(shifted[:,i,j]) == s*s/2:
	# 			mmax[i,j] = True
	# 		if np.argmin(shifted[:,i,j]) == s*s/2:
	# 			mmin[i,j] = True

	return clip(mmin,clip_level),clip(mmax,clip_level)
