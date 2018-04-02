import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener

default_prefs = {
	'fret_min':-.25,
	'fret_max':1.25,
	'fret_nbins':161,

	'hist_type':'stepfilled',
	'hist_color':'steelblue',
	'hist_edgecolor':'black',

	'label_x_nticks':7,
	'filter':False
}

def plot(gui):
	popplot = gui.popout_plots['plot_hist1d'].ui
	popplot.ax[0].cla()
	popplot.resize_fig()
	gui.app.processEvents()

	if gui.ncolors == 2:
		fpb = gui.data.get_plot_data()[0]
		if popplot.prefs['filter'] is True:
			for i in range(fpb.shape[0]):
				cut = np.isfinite(fpb[i])
				try:
					fpb[i][cut] = wiener(fpb[i][cut])
				except:
					pass

		# plt.hist(f.flatten(),bins=181,range=(-.4,1.4),histtype='stepfilled',alpha=.8,normed=True)
		try:
			popplot.ax[0].hist(fpb.flatten(),bins=popplot.prefs['fret_nbins'],range=(popplot.prefs['fret_min'],popplot.prefs['fret_max']),histtype=popplot.prefs['hist_type'],alpha=.8,density=True,color=popplot.prefs['hist_color'],edgecolor=popplot.prefs['hist_edgecolor'])
		except:
			popplot.ax[0].hist(fpb.flatten(),bins=popplot.prefs['fret_nbins'],range=(popplot.prefs['fret_min'],popplot.prefs['fret_max']),histtype='stepfilled',alpha=.8,density=True,color='steelblue')
		if not gui.data.hmm_result is None:
			r = gui.data.hmm_result
			def norm(x,m,v):
				return 1./np.sqrt(2.*np.pi*v)*np.exp(-.5/v*(x-m)**2.)
			def stut(x,a,b,k,m):
				from scipy.special import gammaln
				lam = a*k/(b*(k+1.))
				lny = -.5*np.log(np.pi)
				lny += gammaln((2.*a+1.)/2.) - gammaln((a))
				lny += .5*np.log(lam/(2.*a))
				lny += -.5*(2.*a+1)*np.log(1.+lam*(x-m)**2./(2.*a))
				return np.exp(lny)
				# from scipy.special import gammal
				# v = 2.*a
				# lam = a*k/(b*(k+1.))
				# c = gammaln(.5*(v+1.)) - gammaln(v/2.) - .5*np.log(2.*np.pi/lam)
				# lny = c - .5*(v+1.)*np.log(1.+lam/v*(x-m)**2.)
				# return np.exp(lny)
			x = np.linspace(popplot.prefs['fret_min'],popplot.prefs['fret_max'],1001)
			ppi = np.sum([r.gamma[i].sum(0) for i in range(len(r.gamma))],axis=0)
			ppi /=ppi.sum()
			v = r.b/r.a
			tot = np.zeros_like(x)
			for i in range(r.m.size):
				# y = ppi[i]*norm(x,r.m[i],v[i])
				y = ppi[i]*stut(x,r.a[i],r.b[i],r.beta[i],r.m[i])
				tot += y
				popplot.ax[0].plot(x,y,color='k',lw=1,alpha=.8,ls='--')
			popplot.ax[0].plot(x,tot,color='k',lw=2,alpha=.8)

		popplot.ax[0].set_xlim(popplot.prefs['fret_min'],popplot.prefs['fret_max'])
		popplot.ax[0].set_xticks(np.linspace(popplot.prefs['fret_min'],popplot.prefs['fret_max'],popplot.prefs['label_x_nticks']))
		popplot.ax[0].set_xlabel(r'$\rm E_{\rm FRET}(t)$',fontsize=popplot.prefs['label_fontsize']/gui.plot.canvas.devicePixelRatio())
		popplot.ax[0].set_ylabel('Probability',fontsize=popplot.prefs['label_fontsize']/gui.plot.canvas.devicePixelRatio())
		for asp in ['top','bottom','left','right']:
			popplot.ax[0].spines[asp].set_linewidth(1.0/gui.plot.canvas.devicePixelRatio())
		popplot.f.subplots_adjust(left=.05+popplot.prefs['label_padding'],bottom=.05+popplot.prefs['label_padding'],top=.95,right=.95)
		popplot.f.canvas.draw()
