import numpy as np
import matplolib.pyplot as plt

default_prefs = {
	'plotter_min_fret':-.5,
	'plotter_max_fret':1.5,
	'plotter_nbins_fret':41
}

def plot(gui):
	popplot = gui.docks['plot_hist1d'][1]
	popplot.ax[0].cla()

	if gui.ncolors == 2:
		fpb = gui.data.get_plot_data()[0]

		# plt.hist(f.flatten(),bins=181,range=(-.4,1.4),histtype='stepfilled',alpha=.8,normed=True)
		popplot.ax[0].hist(fpb.flatten(),bins=popplot.prefs['plotter_nbins_fret'],range=(popplot.prefs['plotter_min_fret'],popplot.prefs['plotter_max_fret']),histtype='stepfilled',alpha=.8,normed=True)

		if not gui.data.hmm_result is None:
			r = gui.data.hmm_result
			def norm(x,m,v):
				return 1./np.sqrt(2.*np.pi*v)*np.exp(-.5/v*(x-m)**2.)
			x = np.linspace(popplot.prefs['plotter_min_fret'],popplot.prefs['plotter_max_fret'],1001)
			ppi = np.sum([r.gamma[i].sum(0) for i in range(len(r.gamma))],axis=0)
			ppi /=ppi.sum()
			v = r.b/r.a
			tot = np.zeros_like(x)
			for i in range(r.m.size):
				y = ppi[i]*norm(x,r.m[i],v[i])
				tot += y
				popplot.ax[0].plot(x,y,color='k',lw=1,alpha=.8,ls='--')
			popplot.ax[0].plot(x,tot,color='k',lw=2,alpha=.8)

		popplot.ax[0].set_xlim(popplot.prefs['plotter_min_fret'],popplot.prefs['plotter_max_fret'])
		popplot.ax[0].set_xlabel(r'$\rm E_{\rm FRET}(t)$',fontsize=14./gui.plot.canvas.devicePixelRatio())
		popplot.ax[0].set_ylabel('Probability',fontsize=14./gui.plot.canvas.devicePixelRatio())
		for asp in ['top','bottom','left','right']:
			popplot.ax[0].spines[asp].set_linewidth(1.0/gui.plot.canvas.devicePixelRatio())
		popplot.f.subplots_adjust(left=.13,bottom=.15,top=.95,right=.99)
		popplot.f.canvas.draw()
