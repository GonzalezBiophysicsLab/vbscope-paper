from PyQt5.QtWidgets import QInputDialog
import numpy as np
import matplotlib.pyplot as plt

default_prefs = {
	'synchronize_start_flag':False,
	'plotter_min_time':0,
	'plotter_max_time':100,
	'plotter_nbins_time':100,
	'plotter_min_fret':-.5,
	'plotter_max_fret':1.5,
	'plotter_nbins_fret':41,
	'tau':.1,
	'plotter_2d_syncpreframes':10,
	'plotter_smoothx':.5,
	'plotter_smoothy':.5,
	'plotter_cmap':'rainbow',
	'plotter_floorcolor':'lightgoldenrodyellow',
	'plotter_floor':0.05,
	'plotter_2d_normalizecolumn':False,
	'plotter_timeshift':0,
	'plotter_nbins_contour':20
}

def plot(gui):
	popplot = gui.docks['plot_hist2d'][1]
	popplot.ax[0].cla()#

	if gui.ncolors == 2:
		fpb = gui.data.get_plot_data()[0]
		if popplot.prefs['synchronize_start_flag'] == 'True':
			for i in range(fpb.shape[0]):
				y = fpb[i].copy()
				fpb[i] = np.nan
				pre = gui.data.pre_list[i]
				post = gui.data.pb_list[i]
				if pre < post:
					fpb[i,0:post-pre] = y[pre:post]

		elif not gui.data.hmm_result is None:
			state,success = QInputDialog.getInt(gui,"Pick State","Which State?",min=0,max=gui.data.hmm_result.nstates-1)
			if success:
				v = gui.data.get_viterbi_data()
				vv = np.array([[v[i,:-1],v[i,1:]] for i in range(v.shape[0])])
				oo = []
				for i in range(fpb.shape[0]):
					ms = np.nonzero((vv[i,1]==state)*(vv[i,0]!=vv[i,1]))[0]
					if v[i,0] == state:
						ms = np.append(0,ms)
					ms = np.append(ms,v.shape[1])

					for j in range(ms.size-1):
						o = fpb[i].copy()
						ox = int(np.max((0,ms[j]-popplot.prefs['plotter_2d_syncpreframes'])))
						o = o[ox:ms[j+1]]
						ooo = np.empty(v.shape[1]) + np.nan
						ooo[:o.size] = o
						oo.append(ooo)
				fpb = np.array(oo)


		dtmin = popplot.prefs['plotter_min_time']
		dtmax = popplot.prefs['plotter_max_time']
		if dtmax == -1:
			dtmax = fpb.shape[1]
		dt = np.arange(dtmin,dtmax)*popplot.prefs['tau']
		ts = np.array([dt for _ in range(fpb.shape[0])])
		fpb = fpb[:,dtmin:dtmax]
		xcut = np.isfinite(fpb)
		bt = (popplot.prefs['plotter_max_fret'] - popplot.prefs['plotter_min_fret']) / (popplot.prefs['plotter_nbins_fret'] + 1)
		z,hx,hy = np.histogram2d(ts[xcut],fpb[xcut],bins = [popplot.prefs['plotter_nbins_time'],popplot.prefs['plotter_nbins_fret']+2],range=[[dt[0],dt[-1]],[popplot.prefs['plotter_min_fret']-bt,popplot.prefs['plotter_max_fret']+bt]])
		rx = hx[:-1]
		ry = .5*(hy[1:]+hy[:-1])
		x,y = np.meshgrid(rx,ry,indexing='ij')

		from scipy.ndimage import gaussian_filter
		z = gaussian_filter(z,(popplot.prefs['plotter_smoothx'],popplot.prefs['plotter_smoothy']))

		# cm = plt.cm.rainbow
		# vmin = popplot.prefs['plotter_floor']
		# cm.set_under('w')
		# if vmin <= 1e-300:
		# 	vmin = z.min()
		# pc = plt.pcolor(y.T,x.T,z.T,cmap=cm,vmin=vmin,edgecolors='face')
		try:
			cm = plt.cm.__dict__[popplot.prefs['plotter_cmap']]
		except:
			cm = plt.cm.rainbow
		try:
			cm.set_under(popplot.prefs['plotter_floorcolor'])
		except:
			cm.set_under('w')

		vmin = popplot.prefs['plotter_floor']

		if popplot.prefs['plotter_2d_normalizecolumn'] == 'True':
			z /= np.nanmax(z,axis=1)[:,None]
		else:
			z /= np.nanmax(z)

		z = np.nan_to_num(z)

		x -= popplot.prefs['plotter_timeshift']

		from matplotlib.colors import LogNorm
		if vmin <= 0 or vmin >=z.max():
			pc = popplot.ax[0].contourf(x.T,y.T,z.T,popplot.prefs['plotter_nbins_contour'],cmap=cm)
		else:
			# pc = plt.pcolor(y.T,x.T,z.T,vmin =vmin,cmap=cm,edgecolors='face',lw=1,norm=LogNorm(z.min(),z.max()))
			pc = popplot.ax[0].contourf(x.T,y.T,z.T,popplot.prefs['plotter_nbins_contour'],vmin =vmin,cmap=cm)
		for pcc in pc.collections:
			pcc.set_edgecolor("face")

		try:
			if len(popplot.f.axes) == 1:
				cb = popplot.f.colorbar(pc)
			else:
				popplot.f.axes[1].cla()
				cb = popplot.f.colorbar(pc,cax=popplot.f.axes[1])
			cb.set_ticks(np.array((0.,.2,.4,.6,.8,1.)))
			cb.ax.yaxis.set_tick_params(labelsize=12./gui.plot.canvas.devicePixelRatio(),direction='in',width=1.0/gui.plot.canvas.devicePixelRatio(),length=4./gui.plot.canvas.devicePixelRatio())
			for asp in ['top','bottom','left','right']:
				cb.ax.spines[asp].set_linewidth(1.0/gui.plot.canvas.devicePixelRatio())
			cb.solids.set_edgecolor('face')
			cb.solids.set_rasterized(True)
		except:
			pass

		for asp in ['top','bottom','left','right']:
			popplot.ax[0].spines[asp].set_linewidth(1.0/gui.plot.canvas.devicePixelRatio())
		popplot.f.subplots_adjust(left=.18,bottom=.14,top=.95,right=.99)

		popplot.ax[0].set_xlim(rx.min()-popplot.prefs['plotter_timeshift'],rx.max()-popplot.prefs['plotter_timeshift'])
		popplot.ax[0].set_ylim(popplot.prefs['plotter_min_fret'],popplot.prefs['plotter_max_fret'])
		popplot.ax[0].set_xlabel('Time (s)',fontsize=14./gui.plot.canvas.devicePixelRatio())
		popplot.ax[0].set_ylabel(r'$\rm E_{\rm FRET}(t)$',fontsize=14./gui.plot.canvas.devicePixelRatio())
		bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=1./gui.plot.canvas.devicePixelRatio())
		popplot.ax[0].annotate('n = %d'%(fpb.shape[0]),xy=(.95,.93),xycoords='axes fraction',ha='right',color='k',bbox=bbox_props,fontsize=12./gui.plot.canvas.devicePixelRatio())
		popplot.canvas.draw()
