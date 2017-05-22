import numpy as np
import matplotlib.pyplot as plt
import spotfind as sp
from solid import normal_minmax_dist as nd
from solid import vbem_gmm as vb


d = plt.imread('./pacbio_1_MMStack_Pos0.ome.tif').astype('f')

dx = d.shape[1]/2
d = np.array([d[:,dx*i:dx*(i+1)] for i in range(2)])

coords = []
for side in range(2):
    nsearch = 3
    image = d[side]
    bgg = sp.get_background(image,nsearch*5)
    im = image-bgg

    gmm,locs = sp.classify_spots(im,nstates=5,nrestarts=1,nsearch=nsearch)
    v = sp.spot_cutoff(gmm,0.5,bg_cutoff=.01)

    if 0:

    	hy,hx = plt.hist(im[locs[0],locs[1]],bins=int(locs[0].size**.5),histtype='stepfilled',alpha=.5,normed=True)[:2]
    	x = np.linspace(hx.min(),hx.max(),10000)

    	cut = sp.not_background_class(gmm,.01)
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

    	if 0:
    		f,a=plt.subplots(1)
    		# a[0].imshow(image.T,cmap='viridis',interpolation='nearest',origin='lower')
    		a.imshow(im.T,cmap='viridis',interpolation='nearest',origin='lower')

    		a.plot(locs[0][v],locs[1][v],'o',alpha=.5)
    		a.set_xlim(0,image.shape[0])
    		a.set_ylim(0,image.shape[1])
    		f.tight_layout()
    		plt.show()

    cc = np.array([locs[0][v],locs[1][v]])
    coords.append(cc)

c1,c2 = coords

from icp import icp

transform = icp(c1.T,c2.T)


plt.figure()
plt.plot(c1[0],c1[1],'o')
plt.plot(c2[0],c2[1],'o')
ct2 = transform(c2.T).T
plt.plot(ct2[0],ct2[1],'o')
plt.show()
