import numpy as np
import matplotlib.pyplot as plt
from simulate_image import render
import minmax
from normal_dist import fit_minval_normal,p_normal
from poisson_dist import fit_minval_poisson,p_poisson


def get_background_dist(image,dn=11):
	mmin = minmax.min_map(image,dn)
	mmin = image[mmin]
	mmin = mmin[(mmin > np.percentile(mmin,5))*(mmin < np.percentile(mmin,95))]
	bg = nd.fit_minval_normal(mmin,dn*dn,x0=[mmin.max(),np.var(mmin)])
	return bg

def get_background(image):
	def coarse_background(image,min_r,smooth_r):
		from scipy.ndimage import minimum_filter,gaussian_filter
		bgg = minimum_filter(image,min_r)
		bgg = gaussian_filter(bgg,smooth_r)
		return bgg
	bgg = coarse_background(image,5,5)
	# bgg += coarse_background(image-bgg,10,5)
	# bgg += coarse_background(image-bgg,10,5)
	return bgg


if 1:
	from vbem_fixed import vbem_normal_bg as vb
	import matplotlib.pyplot as plt
	from normal_dist import p_normal
	
	# Make image and get background
	image,noise = render(256,256)
	image += noise
	# image = noise
	bgg = get_background(image)
	image -= (bgg - bgg.mean())

	# get local mins and find background distribution
	nstates = 3
	nsearch_min = 5
	nsearch_max = 5

	mmin = minmax.min_map(image,nsearch_min)
	bgfit = fit_minval_normal((image)[mmin],nsearch_min**2)
	
	mmax = minmax.max_map(image,nsearch_max)
	x = (image)[mmax]

	plt.figure()
	ymin = (image)[mmin]
	ymax = (image)[mmax]
	plt.hist(ymin,bins=100,normed=True,alpha=.5,histtype='stepfilled')
	plt.hist(ymax,bins=100,normed=True,alpha=.5,histtype='stepfilled')

	import normal_dist as nd

	# plt.ylim(0,.08)
	xx = np.linspace(ymin.min(),ymax.max(),1000)
	plt.plot(xx,nd.maxval_normal(xx,nsearch_max**2.,0.,2000.),'k')
	plt.plot(xx,nd.minval_normal(xx,nsearch_min**2.,0.,2000.),'k')
	plt.plot(xx,nd.maxval_normal(xx,nsearch_max**2,*bgfit),'g')
	plt.plot(xx,nd.minval_normal(xx,nsearch_min**2,*bgfit),'b')
	# import poisson_dist as pd
	# bgfit = fit_minval_poisson((image)[mmin],nsearch_min**2,x0=np.array((2000.,2000.)))
	#
	# plt.plot(xx,np.exp(pd.ln_maxval_poisson(xx+bgfit[1],nsearch_max**2,bgfit[0])),'r',lw=2)
	# plt.plot(xx,np.exp(pd.ln_minval_poisson(xx+bgfit[1],nsearch_min**2,bgfit[0])),'r',lw=2)
	# # bgfit = np.array((bgfit[0],bgfit[0]))
	# print i,bgfit
	# plt.plot(xx,np.exp(pd.ln_maxval_poisson(xx+2000,nsearch_max**2,2000)),'r',lw=2)
	# plt.plot(xx,np.exp(pd.ln_minval_poisson(xx+2000,nsearch_min**2,2000)),'r',lw=2)
	
	plt.show()
	def elnpx(aa):
		e_lnp_x = aa.x[:,None]*aa.post.e_mu[None,:]
		e_lnp_x += -.5*(aa.post.e_mu2[None,:] + aa.x[:,None]**2.)
		e_lnp_x *= aa.post.e_lam[None,:]
		e_lnp_x += -.5*np.log(2.*np.pi)
		e_lnp_x += .5*aa.post.e_lnlam[None,:]
		return e_lnp_x
	
	if 1:
		aa = vb(x,nstates,nsearch_max**2,bgfit[0],bgfit[1],init_kmeans=True)
		aa.run(debug=True)
	

	if 1:
		f,a = plt.subplots(1)
		hy,hx = a.hist(x,bins=100,normed=1)[:2]
		xx = np.linspace(image.min(),image.max(),10000)
	
		a.plot(xx,nd.maxval_normal(xx,nsearch_max**2,aa.background.mu, aa.background.var)*aa.pi[0])
		[a.plot(xx,p_normal(xx,aa.post.m[j],aa.post.b[j]/aa.post.a[j])*aa.pi[j]) for j in range(1,nstates+1)]
	
	
		f,a=plt.subplots(1,2)
		a[0].imshow(image.T,cmap='viridis',interpolation='nearest',origin='lower')
		a[1].imshow((image).T,cmap='viridis',interpolation='nearest',origin='lower')
		v = aa.r.argmax(1)
		for i in range(nstates+1):
			a[1].plot(np.nonzero(mmax)[0][v==i],np.nonzero(mmax)[1][v==i],'o',alpha=.5)
		a[1].set_xlim(0,bgg.shape[0])
		a[1].set_ylim(0,bgg.shape[1])
		f.tight_layout()
		plt.show()