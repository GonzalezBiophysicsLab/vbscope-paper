## plot

import numpy as np
import matplotlib.pyplot as plt

def color_normalize_regions(movie, scale_factor=1.):
	# Movie is T,X,Y,Regions
	## take median of last frame for each region and divide the entire movie by that.
	return scale_factor * movie / np.median(movie[-1],axis=(0,1))[None,None,None,:]

def color_scale(movie):
	a = 0.1
	b = 1.
	c = np.median(movie,axis=(0,1))
	d = np.max(movie,axis=(0,1))
	scaled = (movie - c) * ((b-a)/(d-c)) + a
	scaled[scaled < 0.] = 0.
	scaled[scaled > 1.] = 1.
	return scaled

def clim_image(ax,low,high):
	'''
	Input:
	* `ax` is a matplotlib axis with an image (i.e. imshow, contourf)
	* `low`, and `high` are doubles to limit the colormap display at (in dataspace)
	'''
	try:
		ax.images[0].set_clim(low,high)
		ax.figure.canvas.draw()
	except:
		print 'nope'

def imval_nonlinear_percentile(data,percent,scale=10.):
	'''
	input:
	* `data` is an `np.ndarray` of the image
	* `percent` is a double between 0 and 100

	returns:
	* The data value specified by `percent` in a non-linear manner
	'''
	xx = scale*((percent / 100.) -.5)
	y = 1./(1.+np.exp(-xx))
	return np.percentile(data,y*100.)


def scatter_spots(spots,color='r',axis=None):
	if axis is None:
		axis = plt.gca()
	if spots.shape[1] == 2:
		axis.plot(spots[:,0],spots[:,1],'o',color=color,alpha=.5)
	else:
		axis.plot(spots[0],spots[1],'o',color=color,alpha=.5)

def imshow(frame,axis=None):

	if axis is None:
		axis = plt.gca()

	if frame.ndim == 2:
		axis.imshow(frame,cmap='viridis',interpolation='nearest')
	else:
		if frame.shape[-1] == 2:
			# Green Red
			rgb_order = [1,0,2]
			f = np.concatenate((frame,np.zeros_like(frame[...,0])[...,None]),axis=-1)
			axis.imshow(color_scale(f[:,:,rgb_order]),interpolation='nearest')
		elif frame.shape[-1] == 3:
			# Blue Green Red
			rgb_order = [2,1,0]
			axis.imshow(color_scale(frame[:,:,rgb_order]),interpolation='nearest')
