import numpy as np
from .normal_minmax_dist import estimate_from_min
from scipy.ndimage import gaussian_filter

def solve_bg(images,m=5,n=10,sigma=5):
	'''
	Input:
	* images (T,NX,NY)
	* m is pixel space to look in each direction
	* n is time space to look in each direction

	Returns:
	* background (NX,NY)
	'''

	# some useful constants
	mm = 2*m+1
	mx = images.shape[1]/mm
	my = images.shape[2]/mm

	# Find background for image number 'n'
	bg = np.zeros_like(images[n])
	for i in range(mx):
		for j in range(my):
			bb = images[:,i*mm:(i+1)*mm,j*mm:(j+1)*mm].min((1,2))
			bb = estimate_from_min(bb,mm*mm)[0]
			bg[i*mm:(i+1)*mm,j*mm:(j+1)*mm] = bb

	# fill in the edges
	bg[mx*mm:] = bg[mx*mm-1][None,:]
	bg[:,my*mm:] = bg[:,my*mm-1][:,None]

	# smooth the background iamge a little
	bg = gaussian_filter(bg,sigma)

	return bg
