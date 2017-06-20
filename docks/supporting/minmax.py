import numpy as np

def make_shift(image,s):
	grid_x, grid_y = np.mgrid[-(s-1)/2:(s-1)/2+1,-(s-1)/2:(s-1)/2+1]
	grid_x = grid_x.flatten()
	grid_y = grid_y.flatten()
	grid = np.array((grid_x,grid_y)).T

	shifted = np.array([np.roll(np.roll(image,i,axis=0),j,axis=1) for i,j in grid])
	return shifted

def clip(image,s):
	image[:s] = 0
	image[-s:] = 0
	image[:,:s] = 0
	image[:,-s:] = 0
	return image

def max_map(image,s):
	shifted = make_shift(image,s)
	return clip(np.argmax(shifted,axis=0) == s*s/2,s)

def min_map(image,s):
	shifted = make_shift(image,s)
	return clip(np.argmin(shifted,axis=0) == s*s/2,s)

def minmax_map(image,s,clip_level=None):
	if clip_level is None:
		clip_level = s
	shifted = make_shift(image,s)
	mmax = np.argmax(shifted,axis=0) == s*s/2
	mmin = np.argmin(shifted,axis=0) == s*s/2
	# return clip(mmin,s),clip(mmax,s)
	return clip(mmin,clip_level),clip(mmax,clip_level)
