from PyQt5.QtWidgets import QInputDialog
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener

default_prefs = {
	'sync_start':True,
	'sync_postsync':True,
	'time_min':0,
	'time_max':200,
	'time_nbins':200,
	'time_shift':0.0,
	# 'fret_min':-.25,
	# 'fret_max':1.25,
	# 'fret_nbins':81,
	'tau':.1,
	'sync_preframe':10,
	# 'hist_smoothx':.1,
	# 'hist_smoothy':.1,
	# 'color_cmap':'rainbow',
	# 'color_floorcolor':'lightgoldenrodyellow',
	# 'color_floor':0.05,
	# 'color_ceiling':0.95,
	# 'plotter_2d_normalizecolumn':False,
	# 'plotter_nbins_contour':200,

	'sync_hmmstate':0,
	'fig_height':4.,
	'fig_width':4.,
	'label_y_nticks':8,
	'label_x_nticks':5,
	'label_ticksize':8,
	'label_padding_left':.2,
	'label_padding_bottom':.15,
	'label_padding_top':.05,
	'label_padding_right':.05,
	'label_fontsize':12.,
	'label_ticksize':10.,
	'textbox_x':0.95,
	'textbox_y':0.93,
	'textbox_fontsize':10.,
	'textbox_nmol':True,
	'fret_min':-.2,
	'fret_max':1.2,
	'fret_nbins':151,
	'hist_smoothx':2.,
	'hist_smoothy':2.,
	'hist_normalize':True,
	'wiener_filter':False,

	'hist_inerp_res':800,

	'color_cmap':'rainbow',
	'color_floorcolor':'lightgoldenrodyellow',
	'color_ceiling':0.95,
	'color_floor':0.05,
	'color_nticks':5,
}


def gen_histogram(gui,fpb):
	from scipy.ndimage import gaussian_filter
	from scipy.interpolate import interp2d

	popplot = gui.popout_plots['plot_hist2d'].ui

	## make histogram
	dtmin = popplot.prefs['time_min']
	dtmax = popplot.prefs['time_max']
	if dtmin == 0 and dtmax == 0: ## initialize
		dtmax = fpb.shape[1]
		popplot.prefs['time_max'] = dtmax
		popplot.prefs['time_nbins'] = dtmax
		popplot._prefs.update_table()

	if dtmin >= dtmax: ## negative bins
		dtmin = 0
		popplot.prefs['time_min'] = 0
		popplot._prefs.update_table()
	# if dtmax == -1 or dtmax > fpb.shape[1]: ##
		# dtmax = fpb.shape[1]
		# popplot.prefs['time_max'] = dtmax
		# popplot._prefs.update_table()

	x = np.arange(np.max((0,dtmin)),np.min((dtmax,fpb.shape[1])))
	ts = np.array([x for _ in range(fpb.shape[0])])
	fpb = fpb[:,np.max((0,dtmin)):np.min((dtmax,fpb.shape[1]))]
	cut = np.isfinite(fpb)
	y = np.linspace(popplot.prefs['fret_min'],popplot.prefs['fret_max'],popplot.prefs['fret_nbins'])
	z,hx,hy = np.histogram2d(ts[cut],fpb[cut],bins=[popplot.prefs['time_nbins'],y.size],range=[[popplot.prefs['time_min'],popplot.prefs['time_max']],[y.min(),y.max()]])

	x = hx[:-1]#.5*(hx[1:]+hx[:-1])

	# smooth histogram
	z = gaussian_filter(z,(popplot.prefs['hist_smoothx'],popplot.prefs['hist_smoothy']))

	## interpolate histogram - interp2d is backwards...
	f =  interp2d(y,x,z, kind='cubic')
	x = np.linspace(popplot.prefs['time_min'],popplot.prefs['time_max'],popplot.prefs['hist_inerp_res'])#*popplot.prefs['tau']
	y = np.linspace(popplot.prefs['fret_min'],popplot.prefs['fret_max'],popplot.prefs['hist_inerp_res']+1)
	z = f(y,x)
	z[z<0] = 0.


	if popplot.prefs['color_ceiling'] == popplot.prefs['color_floor']:
		popplot.prefs['color_floor'] = 0.0
		popplot.prefs['color_ceiling'] = np.ceil(z.max())
		popplot._prefs.update_table()

	if popplot.prefs['hist_normalize']:
		z /= z.max()

	return x,y,z

def get_data(gui):
	popplot = gui.popout_plots['plot_hist2d'].ui

	fpb = gui.data.get_plot_data()[0]
	if gui.data.hmm_result is None:
		for i in range(fpb.shape[0]):
			y = fpb[i].copy()
			fpb[i] = np.nan
			pre = gui.data.pre_list[i]
			post = gui.data.pb_list[i]
			if pre < post:
				yy = y[pre:post]
				if popplot.prefs['wiener_filter']:
					yy = wiener(yy)
				if popplot.prefs['sync_start'] is True:
					fpb[i,0:post-pre] = yy
				else:
					fpb[i,pre:post] = yy

	else:
		state = popplot.prefs['sync_hmmstate']
		flag = False
		if gui.data.hmm_result.type == 'consensus vbfret':
			if state < gui.data.hmm_result.result.mu.size:
				flag = True
		elif gui.data.hmm_result.type == 'vb' or gui.data.hmm_result.type == 'ml':
			if state < gui.data.hmm_result.results[0].mu.size:
				flag = True
		if flag and popplot.prefs['sync_postsync']:
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
					ox = int(np.max((0,ms[j]-popplot.prefs['sync_preframe'])))
					o = o[ox:ms[j+1]]
					if popplot.prefs['wiener_filter']:
						o = wiener(o)
					ooo = np.empty(v.shape[1]) + np.nan
					ooo[:o.size] = o
					oo.append(ooo)
			fpb = np.array(oo)
	return fpb

def colormap(gui):
	prefs = gui.popout_plots['plot_hist2d'].ui.prefs

	## colormap
	try:
		cm = plt.cm.__dict__[prefs['color_cmap']]
	except:
		cm = plt.cm.rainbow
	try:
		cm.set_under(prefs['color_floorcolor'])
	except:
		cm.set_under('w')
	return cm

def plot(gui):
	popplot = gui.popout_plots['plot_hist2d'].ui
	popplot.ax[0].cla()
	popplot.resize_fig()
	gui.app.processEvents()

	if gui.ncolors != 2 or gui.data.d is None:
		return

	fpb = get_data(gui)
	x,y,z = gen_histogram(gui,fpb)

	### Plotting
	cm = colormap(gui)

	vmin = popplot.prefs['color_floor']
	vmax = popplot.prefs['color_ceiling']

	## imshow is backwards
	tau = popplot.prefs['tau']
	pc = popplot.ax[0].imshow(z.T, cmap=cm, origin='lower',interpolation='none',extent=[x.min()*tau+popplot.prefs['time_shift'],x.max()*tau+popplot.prefs['time_shift'],y.min(),y.max()],aspect='auto',vmin=vmin,vmax=vmax)

	# for pcc in pc.collections:
		# pcc.set_edgecolor("face")

	## Colorbar
	ext='neither'
	if vmin > z.min():
		ext = 'min'
	if len(popplot.f.axes) == 1:
		cb = popplot.f.colorbar(pc,extend=ext)
	else:
		popplot.f.axes[1].cla()
		cb = popplot.f.colorbar(pc,cax=popplot.f.axes[1],extend=ext)

	cbticks = np.linspace(0,vmax,popplot.prefs['color_nticks'])
	cbticks = cbticks[cbticks > popplot.prefs['color_floor']]
	# cbticks = cbticks[cbticks < popplot.prefs['color_ceiling']]
	cbticks = np.append(popplot.prefs['color_floor'], cbticks)
	cbticks = np.append(cbticks, popplot.prefs['color_ceiling'])
	cb.set_ticks(cbticks)
	cb.set_ticklabels(["%.2f"%(cbt) for cbt in cbticks])


	cb.ax.yaxis.set_tick_params(labelsize=popplot.prefs['label_ticksize']/gui.plot.canvas.devicePixelRatio(),direction='in',width=1.0/gui.plot.canvas.devicePixelRatio(),length=4./gui.plot.canvas.devicePixelRatio())
	for asp in ['top','bottom','left','right']:
		cb.ax.spines[asp].set_linewidth(1.0/gui.plot.canvas.devicePixelRatio())
	cb.solids.set_edgecolor('face')
	cb.solids.set_rasterized(True)

	####
	#

	popplot.ax[0].set_yticks(np.linspace(popplot.prefs['fret_min'],popplot.prefs['fret_max'],popplot.prefs['label_y_nticks']))
	popplot.ax[0].set_xticks(np.linspace(popplot.prefs['time_min']*popplot.prefs['tau']+popplot.prefs['time_shift'],popplot.prefs['time_max']*popplot.prefs['tau']+popplot.prefs['time_shift'],popplot.prefs['label_x_nticks']))
	popplot.ax[0].set_xlabel('Time (s)',fontsize=popplot.prefs['label_fontsize']/gui.plot.canvas.devicePixelRatio())
	popplot.ax[0].set_ylabel(r'$\rm E_{\rm FRET}(t)$',fontsize=popplot.prefs['label_fontsize']/gui.plot.canvas.devicePixelRatio())


	bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=1./gui.plot.canvas.devicePixelRatio())
	lstr = 'n = %d'%(fpb.shape[0])
	if popplot.prefs['textbox_nmol']:
		lstr = 'N = %d'%(fpb.shape[0])

	popplot.ax[0].annotate(lstr,xy=(popplot.prefs['textbox_x'],popplot.prefs['textbox_y']),xycoords='axes fraction',ha='right',color='k',bbox=bbox_props,fontsize=popplot.prefs['textbox_fontsize']/gui.plot.canvas.devicePixelRatio())


	for asp in ['top','bottom','left','right']:
		popplot.ax[0].spines[asp].set_linewidth(1.0/gui.plot.canvas.devicePixelRatio())
	popplot.f.subplots_adjust(left=popplot.prefs['label_padding_left'],bottom=popplot.prefs['label_padding_bottom'],top=1.-popplot.prefs['label_padding_top'],right=1.-popplot.prefs['label_padding_right'])

	popplot.f.canvas.draw()
	# cm = colormap(gui)
	#
	#
	# vmin = popplot.prefs['color_floor']
	# vmax = popplot.prefs['color_ceiling']
	#
	#
	# from matplotlib.colors import LogNorm
	# if vmin <= 0 or vmin >=z.max():
	# 	pc = popplot.ax[0].contourf(x.T,y.T,z.T,popplot.prefs['plotter_nbins_contour'],cmap=cm)
	# else:
	# 	# pc = plt.pcolor(y.T,x.T,z.T,vmin =vmin,cmap=cm,edgecolors='face',lw=1,norm=LogNorm(z.min(),z.max()))
	# 	pc = popplot.ax[0].contourf(x.T,y.T,z.T,popplot.prefs['plotter_nbins_contour'],vmin =vmin,vmax=vmax,cmap=cm)
	# for pcc in pc.collections:
	# 	pcc.set_edgecolor("face")
	#
	# try:
	# 	if len(popplot.f.axes) == 1:
	# 		cb = popplot.f.colorbar(pc)
	# 	else:
	# 		popplot.f.axes[1].cla()
	# 		cb = popplot.f.colorbar(pc,cax=popplot.f.axes[1])
	#
	# 	# cbticks = np.linspace(popplot.prefs['color_floor'],popplot.prefs['color_ceiling'],popplot.prefs['label_colorbar_nticks'])
	# 	cbticks = np.linspace(0.,1.,popplot.prefs['label_colorbar_nticks'])
	# 	cbticks = cbticks[cbticks > popplot.prefs['color_floor']]
	# 	cbticks = cbticks[cbticks < popplot.prefs['color_ceiling']]
	# 	cbticks = np.append(popplot.prefs['color_floor'], cbticks)
	# 	cbticks = np.append(cbticks, popplot.prefs['color_ceiling'])
	# 	cb.set_ticks(cbticks)
	# 	cb.set_ticklabels(["%.2f"%(cbt) for cbt in cbticks])
	#
	# 	cb.ax.yaxis.set_tick_params(labelsize=popplot.prefs['label_ticksize']/gui.plot.canvas.devicePixelRatio(),direction='in',width=1.0/gui.plot.canvas.devicePixelRatio(),length=4./gui.plot.canvas.devicePixelRatio())
	# 	for asp in ['top','bottom','left','right']:
	# 		cb.ax.spines[asp].set_linewidth(1.0/gui.plot.canvas.devicePixelRatio())
	# 	cb.solids.set_edgecolor('face')
	# 	cb.solids.set_rasterized(True)
	# except:
	# 	pass
	#
	# for asp in ['top','bottom','left','right']:
	# 	popplot.ax[0].spines[asp].set_linewidth(1.0/gui.plot.canvas.devicePixelRatio())
	# popplot.f.subplots_adjust(left=.05+popplot.prefs['label_padding'],bottom=.05+popplot.prefs['label_padding'],top=.95,right=.95)
	#
	# popplot.ax[0].set_xlim(rx.min()-popplot.prefs['plotter_timeshift'],rx.max()-popplot.prefs['plotter_timeshift'])
	# popplot.ax[0].set_ylim(popplot.prefs['fret_min'],popplot.prefs['fret_max'])
	# popplot.ax[0].set_yticks(np.linspace(popplot.prefs['fret_min'],popplot.prefs['fret_max'],popplot.prefs['label_y_nticks']))
	# popplot.ax[0].set_xticks(np.linspace(popplot.prefs['time_min']*popplot.prefs['tau']-popplot.prefs['plotter_timeshift'],popplot.prefs['time_max']*popplot.prefs['tau']-popplot.prefs['plotter_timeshift'],popplot.prefs['label_x_nticks']))
	# popplot.ax[0].set_xlabel('Time (s)',fontsize=popplot.prefs['label_fontsize']/gui.plot.canvas.devicePixelRatio())
	# popplot.ax[0].set_ylabel(r'$\rm E_{\rm FRET}(t)$',fontsize=popplot.prefs['label_fontsize']/gui.plot.canvas.devicePixelRatio())
	# bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=1./gui.plot.canvas.devicePixelRatio())
	# popplot.ax[0].annotate('n = %d'%(fpb.shape[0]),xy=(popplot.prefs['textbox_x'],popplot.prefs['textbox_y']),xycoords='axes fraction',ha='right',color='k',bbox=bbox_props,fontsize=popplot.prefs['textbox_fontsize']/gui.plot.canvas.devicePixelRatio())
	# popplot.canvas.draw()
