import numpy as np
from simulate_image import render
import minmax as mm

def corr(image,mask,dr=5):
	"""
	Example:
	import matplotlib.pyplot as plt
	import numpy as np
	from simulate_image import render
	import minmax as mm
	
	image = render()
	mmin, mmax = mm.minmax_map(image,s=3)
	corrs = corr(image,mm.clip(mmax,11),dr=11)
	plt.figure()
	tcorr = np.zeros_like(corrs[0][0])
	for i in range(len(corrs)):
		plt.plot(corrs[i][0],corrs[i][1],color='k',alpha=.1,zorder=-1)
		tcorr += corrs[i][1]
	tcorr/=i
	plt.plot(corrs[0][0],tcorr,color='b',alpha=.9,zorder=2)
	plt.show()
	
	"""
	
	grid_x, grid_y = np.mgrid[-(dr-1)/2:(dr-1)/2+1,-(dr-1)/2:(dr-1)/2+1]
	grid_r = np.sqrt(grid_x**2. + grid_y**2.)

	ns = np.nonzero(mask)
	
	corrs = []
	for dxi,dyi in zip(ns[0],ns[1]):
		dx,dy = grid_x+dxi,grid_y+dyi
		dx = dx.flatten().astype('i')
		dy = dy.flatten().astype('i')
		mm = image[dx,dy].mean()
		im0 = (image[dxi,dyi]-mm)
		
		dr = grid_r.flatten()
		drs = np.unique(dr)
	
		corri = np.zeros_like(drs)
		for i in range(drs.size):
			drps = np.nonzero(dr == drs[i])[0]
			corri[i] = np.mean(im0*(image[dx[drps],dy[drps]]-mm))
		corrs.append([drs,corri])
	return corrs


# import matplotlib.pyplot as plt
#
# image = render()
# mmin, mmax = mm.minmax_map(image,s=3)
# corrs = corr(image,mm.clip(mmax,25),dr=25)
# d = np.array(corrs)[:,1]
# x = corrs[0][0]
# from kmeans import kmeans
# nstates = 4
# o = kmeans(d,nstates)
# for i in range(d.shape[0]):
# 	if o.r.argmax(1)[i] != o.mu[:,0].argmin():
# 		plt.plot(x,d[i],'k',alpha=.1)
# for i in range(nstates):
# 	plt.plot(x,o.mu[i],'r',alpha=.9)
# print o.pi
# plt.show()