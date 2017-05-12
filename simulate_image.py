import numpy as np
from normal_dist import p_normal


def render(nx=256,ny=256):
	bin_factor = 1
	noise = 2000.
	psf_var = 1.5*(bin_factor**2.)

	dr = 13*bin_factor

	nxy = np.array((nx,ny))*bin_factor
	image = np.zeros(nxy)

	nspots = np.array((200,100,150))#/2
	counts = np.array((700,1000,800)) * np.sqrt(2.*np.pi * psf_var)
	nmol = nspots.size

	grid_x, grid_y = np.mgrid[-(dr-1)/2:(dr-1)/2+1,-(dr-1)/2:(dr-1)/2+1]
	grid_r = np.sqrt(grid_x**2. + grid_y**2.)

	sinusoid =  np.sin(np.mgrid[1:image.shape[0]+1,1:1+image.shape[1]][0]/100.)**2.
	
	# K x N_k x 2
	positions = [np.random.rand(ns,2)*nxy[None,:] for ns in nspots]

	for k in range(nmol):
		for j in range(nspots[k]):
			dx,dy = grid_x+positions[k][j,0],grid_y+positions[k][j,1]
			cutx = (dx >= 0)*(dx<nxy[0])
			cuty = (dy >= 0)*(dy<nxy[1])
			dx = dx[cutx*cuty].astype('i')
			dy = dy[cutx*cuty].astype('i')
			dr = grid_r[cutx*cuty].astype('i')
			psi = p_normal(dr,0.,psf_var) #* (1.+sinusoid[dx,dy])*10
			photons = np.random.poisson(lam=psi*counts[k]).astype('f')
			image[dx,dy] += photons 

	# Add background noise
	nnoise = np.random.poisson(lam=noise,size=image.shape).astype('f')
	nnoise += np.random.poisson(lam=sinusoid*200.)*np.linspace(0,10,sinusoid.shape[1])[None,:]


	# Bin 
	image = np.sum([image[i::bin_factor] for i in range(bin_factor)],axis=0)
	image = np.sum([image[:,i::bin_factor] for i in range(bin_factor)],axis=0)
	nnoise = np.sum([nnoise[i::bin_factor] for i in range(bin_factor)],axis=0)
	nnoise = np.sum([nnoise[:,i::bin_factor] for i in range(bin_factor)],axis=0)

	
	# image += np.random.rand(*image.shape)
	# nnoise += np.random.rand(*image.shape)
	return image,nnoise-noise