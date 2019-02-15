import numpy as np
import numba as nb

@nb.njit(nogil=True,parallel=True,fastmath=True)
def max_map(data,ni,nj,nk):

	mask = np.ones(data.shape,dtype=nb.boolean)
	nx, ny, nz = data.shape

	for i in range(nx):
		for j in range(ny):
			for k in range(nz):
				dd = data[i,j,k]
				mask[i,j,k] = False
				if i <= nx-ni and i >= ni and j <= ny-nj and j >= nj and k <= nz-nk and k >= nk:
					mask[i,j,k] = True
					for ii in range(max((0,i-ni)), min((i+ni+1,nx))):
						if mask[i,j,k] == False:
							break
						for jj in range(max((0,j-nj)), min((j+nj+1,ny))):
							if mask[i,j,k] == False:
								break
							for kk in range(max((0,k-nk)), min((k+nk+1,nz))):
								if data[ii,jj,kk] >= dd:
									if not (ii==i and jj==j and kk==k):
										mask[i,j,k] = False
										break
	return mask

@nb.njit(nogil=True,parallel=True,fastmath=True)
def min_map(data,ni,nj,nk):

	mask = np.ones(data.shape,dtype=nb.boolean)
	nx, ny, nz = data.shape

	for i in range(nx):
		for j in range(ny):
			for k in range(nz):
				dd = data[i,j,k]
				mask[i,j,k] = False
				if i <= nx-ni and i >= ni and j <= ny-nj and j >= nj and k <= nz-nk and k >= nk:
					mask[i,j,k] = True
					for ii in range(max((0,i-ni)), min((i+ni+1,nx))):
						if mask[i,j,k] == False:
							break
						for jj in range(max((0,j-nj)), min((j+nj+1,ny))):
							if mask[i,j,k] == False:
								break
							for kk in range(max((0,k-nk)), min((k+nk+1,nz))):
								if data[ii,jj,kk] <= dd:
									if not (ii==i and jj==j and kk==k):
										mask[i,j,k] = False
										break
	return mask

def minmax_map(data,ni,nj,nk,nclip):
	minmap = min_map(data,ni,nj,nk)
	maxmap = max_map(data,ni,nj,nk)
	minmap = clip_edges(minmap,(0,nclip,nclip))
	maxmap = clip_edges(maxmap,(0,nclip,nclip))
	return minmap,maxmap

def _get_shape(data,footprint):
	## footprint is a tuple of dimensions in data
	shape = None
	if isinstance(footprint,int):
		shape = [footprint]*data.ndim
	elif isinstance(footprint,tuple) or isinstance(footprint,list):
		if len(footprint) == data.ndim:
			shape = footprint
	return shape
def clip_edges(data,footprint,value=0):
	shape = _get_shape(data,footprint)
	if shape is None:
		logging.warning('Footprint is bad')
		return data

	for i in range(len(shape)):
		s = [slice(0,data.shape[i]) for i in range(len(shape))]
		s[i] = slice(0,shape[i])
		data[tuple(s)] = value
		s[i] = slice(data.shape[i]-shape[i],data.shape[i])
		data[tuple(s)] = value
	return data

#
# @nb.njit
# def make_shift(image,s):
# 	nx,ny = image.shape
# 	ds = (s-1)//2
# 	shifted = np.zeros((s*s,nx,ny))
#
# 	for i in range(nx):
# 		for j in range(ny):
# 			for ii in range(-ds,ds+1):
# 				for jj in range(-ds,ds+1):
# 					if ii + i >= 0 and ii + i < nx and jj + j >= 0 and jj + j < ny:
# 						shifted[(ds+ii)*s+(ds+jj),i,j] = image[ii+i,jj+j]
# 	return shifted
#
@nb.njit
def clip(image,s):
	iimage = np.copy(image)
	iimage[:s] = 0
	iimage[-s:] = 0
	iimage[:,:s] = 0
	iimage[:,-s:] = 0
	return iimage
#
# def max_map(image,s):
# 	shifted = make_shift(image,s)
# 	middle = (s+1)*(s-1)/2
# 	return clip(np.argmax(shifted,axis=0) == middle,s)
#
# def min_map(image,s):
# 	shifted = make_shift(image,s)
# 	middle = (s+1)*(s-1)/2
# 	return clip(np.argmin(shifted,axis=0) == middle,s)
#
# # @nb.njit
# def minmax_map(image,s,clip_level=None):
# 	if clip_level is None:
# 		clip_level = s
#
# 	mmax = max_map(image,s)
# 	mmin = min_map(image,s)
# 	# shifted = make_shift(image,s)
# 	#
# 	# mmax = np.zeros_like(image,dtype='bool')
# 	# mmin = np.zeros_like(image,dtype='bool')
# 	# for i in range(image.shape[0]):
# 	# 	for j in range(image.shape[1]):
# 	# 		if np.argmax(shifted[:,i,j]) == s*s/2:
# 	# 			mmax[i,j] = True
# 	# 		if np.argmin(shifted[:,i,j]) == s*s/2:
# 	# 			mmin[i,j] = True
#
# 	return clip(mmin,clip_level),clip(mmax,clip_level)
