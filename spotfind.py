import numpy as np
import matplotlib.pyplot as plt
from simulate_image import render
from solid import minmax
from solid import normal_minmax_dist as nd
from solid import vbem_gmm as vb


def get_background_dist(image,dn=11):
	mmin = minmax.min_map(image,dn)
	mmin = image[mmin]
	mmin = mmin[(mmin > np.percentile(mmin,5))*(mmin < np.percentile(mmin,95))]
	bg = nd.fit_minval_normal(mmin,dn*dn,x0=[mmin.max(),np.var(mmin)])
	return bg

def get_background(image,rad=5):
	from scipy.ndimage import median_filter,gaussian_filter,minimum_filter
	# def coarse_background(image,min_r,smooth_r):
	# 	bgg = minimum_filter(image,min_r)
	# 	bgg = gaussian_filter(bgg,smooth_r)
	# 	return bgg
	# bgg = coarse_background(image,5,5)
	# bgg += coarse_background(image-bgg,10,5)
	# bgg += coarse_background(image-bgg,10,5)
	bgg = gaussian_filter(median_filter(image,rad),rad/10.)
	# bgg = np.fft.fft2(image)
	return bgg



def classify_spots(image,nsearch=3,nstates=5,nthreads=4,nrestarts=3):
	# nstates = 5
	# nsearch = 5
	# nrestarts = 3

	## Find local mins and local maxes
	mmin,mmax = minmax.minmax_map(image,nsearch)

	## Estimate background distribution from local mins
	bgfit = nd.estimate_from_min(image[mmin],nsearch**2)
	background = vb.background(nsearch**2,*bgfit)

	## Classify local maxes
	gmm = vb.robust_vbem(image[mmax],nstates,background,nrestarts=nrestarts,nthreads=nthreads)

	return gmm,np.nonzero(mmax)

def not_background_class(aa,cutoff=.001):
	l = np.arange(aa.post.m.size)
	if aa._bg_flag:
		p_bg = np.exp(aa.background.lnprob(aa.post.m)) # prob of being max-val background
		p = np.exp(aa.background.lnprob(aa.background.e_max_m))
		cut = (aa.post.m > aa.background.e_max_m)*(p_bg < p*cutoff)
		return l[cut]
	return l

def spot_cutoff(gmm,p_cutoff,bg_cutoff=0.001,class_list=None):
	if class_list is None:
		class_list = not_background_class(gmm,bg_cutoff)


	probs = (gmm.r[:,class_list]).sum(1)/gmm.r.sum(1)
	spots = probs > p_cutoff

	return spots


if 0:
	import matplotlib.pyplot as plt

	# Make image and get background
	image,noise = render(256,512)

	# image += 1e-300*np.random.rand(*image.shape)
	image += noise
	# image = noise

	# bgg = get_background(image)
	# image -= bgg

	nsearch = 3
	bgg = get_background(image,nsearch*5)
	im = image-bgg
	gmm,locs = classify_spots(im,nstates=10,nrestarts=4,nsearch=nsearch)
	v = spot_cutoff(gmm,0.5)



	if 1:

		hy,hx = plt.hist(im[locs[0],locs[1]],bins=int(locs[0].size**.5),histtype='stepfilled',alpha=.5,normed=True)[:2]
		x = np.linspace(hx.min(),hx.max(),10000)

		cut = not_background_class(gmm,.01)
		# i = 1
		# print nd.p_normal(x,gmm.prior.m[i],1./gmm.prior.beta[i]).sum()
		[plt.plot(x,nd.p_normal(x,gmm.prior.m[i],1./gmm.prior.beta[i]),'g',ls='--') for i in cut]
		[plt.plot(x,nd.p_normal(x,gmm.prior.m[i],gmm.prior.b[i]/gmm.prior.a[i]),'k',ls='--') for i in cut]
		[plt.plot(x,nd.p_normal(x,gmm.post.m[i],gmm.post.b[i]/gmm.post.a[i])*gmm.pi[i],'k',ls='-') for i in cut]
		plt.plot(x,np.array([nd.p_normal(x,gmm.post.m[i],gmm.post.b[i]/gmm.post.a[i])*gmm.pi[i] for i in cut]).sum(0),'b',ls='-')

		print gmm.background.mu,gmm.background.var
		print gmm.background.e_max_m,gmm.background.e_max_var
		plt.plot(x,nd.p_normal_max(x,nsearch**2,gmm.background.mu, gmm.background.var)*gmm.pi[0],'r')

		# plt.yscale('log')
		# plt.ylim(hy[hy != 0].min(),hy[hy!=0.].max())
		# plt.show()

		plt.show()

	if 1:
		f,a=plt.subplots(1)
		# a[0].imshow(image.T,cmap='viridis',interpolation='nearest',origin='lower')
		a.imshow(im.T,cmap='viridis',interpolation='nearest',origin='lower')

		a.plot(locs[0][v],locs[1][v],'o',alpha=.5)
		a.set_xlim(0,image.shape[0])
		a.set_ylim(0,image.shape[1])
		f.tight_layout()
		plt.show()

	if 1:
		plt.hist(image.flatten(),bins=200,histtype='stepfilled',log=True,alpha=.5)
		plt.hist(image[locs[0][v],locs[1][v]],bins=30,histtype='stepfilled',log=True,alpha=.5)
		plt.show()

	# if 0:
	# 	plt.figure()
	# 	ymin = (image)[mmin]
	# 	ymax = (image)[mmax]
	# 	hy1,hx1 = plt.hist(ymin,bins=100,normed=True,alpha=.5,histtype='stepfilled',log=True)[:2]
	# 	hy2,hx2 = plt.hist(ymax,bins=100,normed=True,alpha=.5,histtype='stepfilled',log=True)[:2]
	#
	# 	xx = np.linspace(ymin.min(),ymax.max(),1000)
	# 	# plt.plot(xx,nd.p_normal_max(xx,nsearch_max**2.,0.,2000.),'k')
	# 	# plt.plot(xx,nd.p_normal_min(xx,nsearch_min**2.,0.,2000.),'k')
	# 	plt.plot(xx,nd.p_normal_max(xx,nsearch_max**2,*bgfit))
	# 	plt.plot(xx,nd.p_normal_min(xx,nsearch_min**2,*bgfit))
	# 	plt.ylim(hy1[hy1 != 0].min(),hy1[hy1!=0.].max())
	# 	# import poisson_dist as pd
	# 	# bgfit = fit_minval_poisson((image)[mmin],nsearch_min**2,x0=np.array((2000.,2000.)))
	# 	#
	# 	# plt.plot(xx,np.exp(pd.ln_maxval_poisson(xx+bgfit[1],nsearch_max**2,bgfit[0])),'r',lw=2)
	# 	# plt.plot(xx,np.exp(pd.ln_minval_poisson(xx+bgfit[1],nsearch_min**2,bgfit[0])),'r',lw=2)
	# 	# # bgfit = np.array((bgfit[0],bgfit[0]))
	# 	# print i,bgfit
	# 	# plt.plot(xx,np.exp(pd.ln_maxval_poisson(xx+2000,nsearch_max**2,2000)),'r',lw=2)
	# 	# plt.plot(xx,np.exp(pd.ln_minval_poisson(xx+2000,nsearch_min**2,2000)),'r',lw=2)
	#
	# 	plt.show()
	# 	# def elnpx(aa):
	# 	# 	e_lnp_x = aa.x[:,None]*aa.post.e_mu[None,:]
	# 	# 	e_lnp_x += -.5*(aa.post.e_mu2[None,:] + aa.x[:,None]**2.)
	# 	# 	e_lnp_x *= aa.post.e_lam[None,:]
	# 	# 	e_lnp_x += -.5*np.log(2.*np.pi)
	# 	# 	e_lnp_x += .5*aa.post.e_lnlam[None,:]
	# 	# 	return e_lnp_x
	#
	# #
	#
	#
	# if 1:
	# 	f,a = plt.subplots(1)
	# 	hy,hx = a.hist(x,bins=100,normed=1,histtype='stepfilled',alpha=.5)[:2]
	# 	xx = np.linspace(image.min(),image.max(),10000)
	#
	#
	# 	cut = not_background_list(aa)
	# 	print cut
	# 	# cut = np.arange(aa.post.m.size)[1:]
	# 	# cut = cut[aa.post.m > aa.background.e_max_m]
	#
	# 	a.plot(xx,nd.p_normal_max(xx,nsearch_max**2,aa.background.mu, aa.background.var)*aa.pi[0],'r')
	# 	# [a.plot(xx,nd.p_normal(xx,aa.post.m[j],aa.post.b[j]/aa.post.a[j])*aa.pi[j]) for j in range(nstates+1)]
	# 	print aa.prior.m
	# 	print aa.post.m
	# 	print aa.pi
	# 	[plt.plot(xx,nd.p_normal(xx,aa.prior.m[i],1./aa.prior.beta[i]),'g',ls='--') for i in cut]
	# 	[plt.plot(xx,nd.p_normal(xx,aa.prior.m[i],aa.prior.b[i]/aa.prior.a[i]),'k',ls='--') for i in cut]
	# 	[plt.plot(xx,nd.p_normal(xx,aa.post.m[i],aa.post.b[i]/aa.post.a[i])*aa.pi[i],'k',ls='-') for i in cut]
	# 	plt.plot(xx,np.array([nd.p_normal(xx,aa.post.m[i],aa.post.b[i]/aa.post.a[i])*aa.pi[i] for i in cut]).sum(0),'b',ls='-')
	#
	# 	plt.yscale('log')
	# 	plt.ylim(hy[hy != 0].min(),hy[hy!=0.].max())
	# 	# plt.show()
	#
	# if 1:
	# 	f,a=plt.subplots(1,2)
	# 	a[0].imshow(image.T,cmap='viridis',interpolation='nearest',origin='lower')
	# 	a[1].imshow((image).T,cmap='viridis',interpolation='nearest',origin='lower')
	# 	v = aa.r.argmax(1)
	# 	v = spots_prob_notbg(aa,cut) > .5
	# 	for i in [1]:
	# 		a[1].plot(np.nonzero(mmax)[0][v==i],np.nonzero(mmax)[1][v==i],'o',alpha=.5)
	# 	a[1].set_xlim(0,bgg.shape[0])
	# 	a[1].set_ylim(0,bgg.shape[1])
	# 	f.tight_layout()
	# plt.show()
