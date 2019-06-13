import numpy as np
import matplotlib.pyplot as plt

default_prefs = {
	'hist_type':'stepfilled',
	'hist_color':'steelblue',
	'hist_edgecolor':'black',
	'hist_nbins':101,

	'label_x_nticks':5,

	'intensity_min':-1000,
	'intensity_max':10000,

	'hist_logy':True,

	'fig_width':9.0,
	'fig_height':3.0
}

def plot(gui):
	popplot = gui.popout_plots['plot_intensities'].ui
	if popplot.ax.size != gui.ncolors + 1:
		popplot.nplots_y = gui.ncolors + 1
		popplot.prefs['fig_width'] = popplot.prefs['fig_height'] * (gui.ncolors+1)
	popplot.resize_fig()
	gui.app.processEvents()

	d = gui.data.get_fluor() ## N,C,D

	loghist = True if (popplot.prefs['hist_logy'] is True) else False
	nbins = popplot.prefs['hist_nbins']
	from scipy.stats import gaussian_kde
	dd = d.sum(-1)

	x = np.linspace(dd.min(),dd.max(),10000)
	for i in range(gui.ncolors):
		# nbins = int(np.ceil(2 * dd.shape[0]**(1./3.)))
		hy,hx = popplot.ax[i].hist(dd[:,i],range=(dd.min(),dd.max()),bins=nbins,histtype=popplot.prefs['hist_type'],alpha=.8,density=True,color=popplot.prefs['hist_color'],edgecolor=popplot.prefs['hist_edgecolor'],log=loghist)[:2]
		kernel = gaussian_kde(dd[:,i])
		ylim = popplot.ax[i].get_ylim()
		popplot.ax[i].plot(x,kernel(x),color='r',alpha=.8,lw=1)
		popplot.ax[i].set_ylim(*ylim)

	# nbins = int(np.ceil(2 * dd.shape[0]**(1./3.)))
	hy,hx = popplot.ax[-1].hist(dd.sum(1),range=(dd.min(),dd.max()),bins=nbins,histtype=popplot.prefs['hist_type'],alpha=.8,density=True,color=popplot.prefs['hist_color'],edgecolor=popplot.prefs['hist_edgecolor'],log=loghist)[:2]
	kernel = gaussian_kde(dd.sum(1))
	ylim = popplot.ax[-1].get_ylim()
	popplot.ax[-1].plot(x,kernel(x),color='r',alpha=.8,lw=1)
	popplot.ax[-1].set_ylim(*ylim)
	#
	# m0s,v0s,m1s,v1s,fracs = gui.data.estimate_mvs()
	# gui.data.posterior_sum()
	#
	# for i in range(gui.ncolors):
	# 	dd = d[:,i]
	# 	hy,hx = popplot.ax[i+1].hist(dd.flatten(),range=(popplot.prefs['intensity_min'],popplot.prefs['intensity_max']),bins=popplot.prefs['hist_nbins'],histtype=popplot.prefs['hist_type'],alpha=.8,density=True,color=popplot.prefs['hist_color'],edgecolor=popplot.prefs['hist_edgecolor'],log=loghist)[:2]
	#
	# 	x = np.linspace(hx.min(),hx.max(),10000)
	#
	# 	ylim = popplot.ax[i+1].get_ylim()
	# 	y1 = normal(x,m0s[i],v0s[i])*fracs[2*i+0]
	# 	y2 = normal(x,m1s[i],v1s[i])*fracs[2*i+1]
	# 	popplot.ax[i+1].plot(x,y1,color='black',alpha=.8,lw= 1)
	# 	popplot.ax[i+1].plot(x,y2,color='black',alpha=.8,lw= 1)
	# 	popplot.ax[i+1].plot(x,y1+y2,color='black',alpha=.8,lw= 1)
	# 	popplot.ax[i+1].set_ylim(*ylim)
	#
	# dd = d.sum(1)
	# hy,hx = popplot.ax[0].hist(dd.flatten(),bins=popplot.prefs['hist_nbins'],range=(popplot.prefs['intensity_min'],popplot.prefs['intensity_max']),histtype=popplot.prefs['hist_type'],alpha=.8,density=True,color=popplot.prefs['hist_color'],edgecolor=popplot.prefs['hist_edgecolor'],log=loghist)[:2]
	#
	# x = np.linspace(hx.min(),hx.max(),10000)
	#
	# ylim = popplot.ax[0].get_ylim()
	# y1 = normal(x,m0s[-1],v0s[-1])*fracs[-2]
	# y2 = normal(x,m1s[-1],v1s[-1])*fracs[-1]
	# popplot.ax[0].plot(x,y1,color='black',alpha=.8,lw= 1)
	# popplot.ax[0].plot(x,y2,color='black',alpha=.8,lw= 1)
	# popplot.ax[0].plot(x,y1+y2,color='black',alpha=.8,lw= 1)
	# popplot.ax[0].set_ylim(*ylim)
	#
	for aa in popplot.ax[1:]:
		aa.set_yticks(())

	for i in range(len(popplot.ax)):
		popplot.ax[i].ticklabel_format(scilimits=(-4,4),axis='x')
		popplot.ax[i].set_title('Color %d'%(i))
	popplot.ax[-1].set_title('Color 0+1')
	# for aa in popplot.ax:
		# aa.set_xlim(popplot.prefs['intensity_min'],popplot.prefs['intensity_max'])
	#
	#
	#
	#
	#
	#
	#
	# # 	fpb = gui.data.get_plot_data()[0]
	# #
	# # 	# plt.hist(f.flatten(),bins=181,range=(-.4,1.4),histtype='stepfilled',alpha=.8,normed=True)
	# # 	try:
	# # 		popplot.ax[0].hist(fpb.flatten(),bins=popplot.prefs['fret_nbins'],range=(popplot.prefs['fret_min'],popplot.prefs['fret_max']),histtype=popplot.prefs['hist_type'],alpha=.8,density=True,color=popplot.prefs['hist_color'],edgecolor=popplot.prefs['hist_edgecolor'])
	# # 	except:
	# # 		popplot.ax[0].hist(fpb.flatten(),bins=popplot.prefs['fret_nbins'],range=(popplot.prefs['fret_min'],popplot.prefs['fret_max']),histtype='stepfilled',alpha=.8,density=True,color='steelblue')
	# # 	if not gui.data.hmm_result is None:
	# # 		r = gui.data.hmm_result
	# # 		def norm(x,m,v):
	# # 			return 1./np.sqrt(2.*np.pi*v)*np.exp(-.5/v*(x-m)**2.)
	# # 		x = np.linspace(popplot.prefs['fret_min'],popplot.prefs['fret_max'],1001)
	# # 		ppi = np.sum([r.gamma[i].sum(0) for i in range(len(r.gamma))],axis=0)
	# # 		ppi /=ppi.sum()
	# # 		v = r.b/r.a
	# # 		tot = np.zeros_like(x)
	# # 		for i in range(r.m.size):
	# # 			y = ppi[i]*norm(x,r.m[i],v[i])
	# # 			tot += y
	# # 			popplot.ax[0].plot(x,y,color='k',lw=1,alpha=.8,ls='--')
	# # 		popplot.ax[0].plot(x,tot,color='k',lw=2,alpha=.8)
	# #
	# # 	popplot.ax[0].set_xlim(popplot.prefs['fret_min'],popplot.prefs['fret_max'])
	# # 	popplot.ax[0].set_xticks(np.linspace(popplot.prefs['fret_min'],popplot.prefs['fret_max'],popplot.prefs['label_x_nticks']))
	# # 	popplot.ax[0].set_xlabel(r'$\rm E_{\rm FRET}(t)$',fontsize=popplot.prefs['label_fontsize']/gui.plot.canvas.devicePixelRatio())
	# # 	popplot.ax[0].set_ylabel('Probability',fontsize=popplot.prefs['label_fontsize']/gui.plot.canvas.devicePixelRatio())
	# # 	for asp in ['top','bottom','left','right']:
	# # 		popplot.ax[0].spines[asp].set_linewidth(1.0/gui.plot.canvas.devicePixelRatio())
	# # 	popplot.f.subplots_adjust(left=.05+popplot.prefs['label_padding'],bottom=.05+popplot.prefs['label_padding'],top=.95,right=.95)


	popplot.f.tight_layout()
	popplot.f.canvas.draw()


def normal(x,m,v):
	return 1./np.sqrt(2.*np.pi*v) * np.exp(-.5/v*(x-m)**2.)
