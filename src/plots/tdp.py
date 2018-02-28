import numpy as np
import matplotlib.pyplot as plt

default_prefs = {
'plotter_min_fret':-.5,
'plotter_max_fret':1.5,
'plotter_nbins_fret':41,
'plotter_smoothx':0.5,
'plotter_smoothy':0.5,
'plotter_cmap':'rainbow',
'plotter_floor':1,
'plotter_floorcolor':'lightgoldenrodyellow',
'plotter_nbins_contour':20
}

def plot(gui):
	popplot = gui.docks['plot_tdp'][1]
	popplot.ax[0].cla()

	if gui.ncolors == 2:
		fpb = gui.data.get_plot_data()[0]
		d = np.array([[fpb[i,:-1],fpb[i,1:]] for i in range(fpb.shape[0])])

		if not gui.data.hmm_result is None:
			# state,success = QInputDialog.getInt(self,"Pick State","Which State?",min=0,max=gui.data.hmm_result.nstates-1)
			# if success:
			v = gui.data.get_viterbi_data()
			vv = np.array([[v[i,:-1],v[i,1:]] for i in range(v.shape[0])])

			for i in range(d.shape[0]):
				d[i,:,vv[i,0]==vv[i,1]] = np.array((np.nan,np.nan))


		print d.shape
		rx = np.linspace(popplot.prefs['plotter_min_fret'],popplot.prefs['plotter_max_fret'],popplot.prefs['plotter_nbins_fret'])
		ry = np.linspace(popplot.prefs['plotter_min_fret'],popplot.prefs['plotter_max_fret'],popplot.prefs['plotter_nbins_fret'])
		x,y = np.meshgrid(rx,ry,indexing='ij')
		dx = d[:,0].flatten()
		dy = d[:,1].flatten()
		cut = np.isfinite(dx)*np.isfinite(dy)
		z,hx,hy = np.histogram2d(dx[cut],dy[cut],bins=[rx.size,ry.size],range=[[rx.min(),rx.max()],[ry.min(),ry.max()]])

		from scipy.ndimage import gaussian_filter
		z = gaussian_filter(z,(popplot.prefs['plotter_smoothx'],popplot.prefs['plotter_smoothy']))

		try:
			cm = plt.cm.__dict__[popplot.prefs['plotter_cmap']]
		except:
			cm = plt.cm.rainbow
		try:
			cm.set_under(popplot.prefs['plotter_floorcolor'])
		except:
			cm.set_under('w')

		from matplotlib.colors import LogNorm
		if popplot.prefs['plotter_floor'] <= 0:
			bins = np.logspace(np.log10(z[z>0.].min()),np.log10(z.max()),popplot.prefs['plotter_nbins_contour'])
			pc = popplot.ax[0].contourf(x, y, z, bins, cmap=cm, norm=LogNorm())
		else:
			z[z< 1e-10] = 1e-9
			bins = np.logspace(0,np.log10(z.max()),popplot.prefs['plotter_nbins_contour'])
			bins = np.append(1e-10,bins)
			pc = popplot.ax[0].contourf(x, y, z, bins, vmin=popplot.prefs['plotter_floor'], cmap=cm, norm=LogNorm())

		for pcc in pc.collections:
			pcc.set_edgecolor("face")

		if len(popplot.f.axes) == 1:
			cb = popplot.f.colorbar(pc)
		else:
			popplot.f.axes[1].cla()
			cb = popplot.f.colorbar(pc,cax=popplot.f.axes[1])
		zm = np.floor(np.log10(z.max()))
		cz = np.logspace(0,zm,zm+1)
		cb.set_ticks(cz)
		# cb.set_ticklabels(cz)
		cb.ax.yaxis.set_tick_params(labelsize=12./gui.plot.canvas.devicePixelRatio(),direction='in',width=1.0/gui.plot.canvas.devicePixelRatio(),length=4./gui.plot.canvas.devicePixelRatio())
		for asp in ['top','bottom','left','right']:
			cb.ax.spines[asp].set_linewidth(1.0/gui.plot.canvas.devicePixelRatio())
		cb.solids.set_edgecolor('face')
		cb.solids.set_rasterized(True)

		popplot.ax[0].set_xlim(rx.min(),rx.max())
		popplot.ax[0].set_ylim(ry.min(),ry.max())
		popplot.ax[0].set_xlabel(r'Initial E$_{\rm FRET}$',fontsize=14./gui.plot.canvas.devicePixelRatio())
		popplot.ax[0].set_ylabel(r'Final E$_{\rm FRET}$',fontsize=14./gui.plot.canvas.devicePixelRatio())
		popplot.ax[0].set_title('Transition Density (Counts)',fontsize=12/gui.plot.canvas.devicePixelRatio())
		bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=1./gui.plot.canvas.devicePixelRatio())
		popplot.ax[0].annotate('n = %d'%(fpb.shape[0]),xy=(.95,.93),xycoords='axes fraction',ha='right',color='k',bbox=bbox_props,fontsize=12./gui.plot.canvas.devicePixelRatio())

		for asp in ['top','bottom','left','right']:
			popplot.ax[0].spines[asp].set_linewidth(1.0/gui.plot.canvas.devicePixelRatio())
		popplot.f.subplots_adjust(left=.18,bottom=.14,top=.92,right=.99)

		popplot.f.canvas.draw()
