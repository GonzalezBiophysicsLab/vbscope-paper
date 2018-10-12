from PyQt5.QtWidgets import QInputDialog
import numpy as np
import matplotlib.pyplot as plt


default_prefs = {
	'subplots_top':0.97,
	'axes_topright':True,
	'xlabel_offset':-0.1,
	'ylabel_offset':-0.18,
	'fig_height':2.5,

	'time_min':0,
	'time_max':200,
	'time_nbins':200,
	'time_shift':0.0,
	'time_nticks':5,
	'time_dt':1.,

	'sync_start':True,
	'sync_postsync':True,
	'sync_preframe':10,
	'sync_hmmstate':0,

	'textbox_x':0.97,
	'textbox_y':0.92,
	'textbox_fontsize':8.,
	'textbox_nmol':True,

	'fret_min':-.25,
	'fret_max':1.25,
	'fret_nbins':51,
	'fret_nticks':7,

	'hist_smoothx':1.,
	'hist_smoothy':1.,
	'hist_normalize':True,
	'hist_inerp_res':800,

	'filter':False,

	'color_cmap':'rainbow',
	'color_floorcolor':'lightgoldenrodyellow',
	'color_ceiling':0.95,
	'color_floor':0.05,
	'color_nticks':5,

	'xlabel_text':r'Time (s)',
	'ylabel_text':r'E$_{\rm{FRET}}$'
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

	if dtmin >= dtmax: ## negative bins
		dtmin = 0
		popplot.prefs['time_min'] = 0
	# if dtmax == -1 or dtmax > fpb.shape[1]: ##
		# dtmax = fpb.shape[1]
		# popplot.prefs['time_max'] = dtmax

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
	x = np.linspace(popplot.prefs['time_min'],popplot.prefs['time_max'],popplot.prefs['hist_inerp_res'])#*popplot.prefs['time_dt']
	y = np.linspace(popplot.prefs['fret_min'],popplot.prefs['fret_max'],popplot.prefs['hist_inerp_res']+1)
	z = f(y,x)
	z[z<0] = 0.


	if popplot.prefs['color_ceiling'] == popplot.prefs['color_floor']:
		popplot.prefs['color_floor'] = 0.0
		popplot.prefs['color_ceiling'] = np.ceil(z.max())

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
				if popplot.prefs['filter']:
					yy = gui.data.filter(yy)
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
					if popplot.prefs['filter']:
						o = gui.data.filter(o)
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
	if gui.data.d is None:
		return
	popplot = gui.popout_plots['plot_hist2d'].ui
	pp = popplot.prefs
	popplot.ax[0].cla()
	popplot.resize_fig()
	gui.app.processEvents()

	dpr = popplot.f.canvas.devicePixelRatio()

	if gui.ncolors != 2 or gui.data.d is None:
		return

	fpb = get_data(gui)
	x,y,z = gen_histogram(gui,fpb)

	### Plotting
	cm = colormap(gui)

	vmin = pp['color_floor']
	vmax = pp['color_ceiling']

	## imshow is backwards
	tau = pp['time_dt']
	pc = popplot.ax[0].imshow(z.T, cmap=cm, origin='lower',interpolation='none',extent=[x.min()*tau+pp['time_shift'],x.max()*tau+pp['time_shift'],y.min(),y.max()],aspect='auto',vmin=vmin,vmax=vmax)

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

	cbticks = np.linspace(0,vmax,pp['color_nticks'])
	cbticks = np.array(popplot.figure_out_ticks(0,pp['color_ceiling'],pp['color_nticks']))
	cbticks = cbticks[cbticks > pp['color_floor']]
	cbticks = cbticks[cbticks < popplot.prefs['color_ceiling']]
	cbticks = np.append(pp['color_floor'], cbticks)
	cbticks = np.append(cbticks, pp['color_ceiling'])
	cb.set_ticks(cbticks)
	cb.set_ticklabels(["%.2f"%(cbt) for cbt in cbticks])
	for label in cb.ax.get_yticklabels():
		label.set_family(pp['font'])

	cb.ax.yaxis.set_tick_params(labelsize=pp['tick_fontsize']/dpr,direction=pp['tick_direction'],width=pp['tick_linewidth']/dpr,length=pp['tick_length_major']/dpr)
	for asp in ['top','bottom','left','right']:
		cb.ax.spines[asp].set_linewidth(pp['axes_linewidth']/dpr)
	cb.solids.set_edgecolor('face')
	cb.solids.set_rasterized(True)

	####################################################
	####################################################

	popplot.ax[0].set_xticks(popplot.figure_out_ticks(pp['time_min']*pp['time_dt']+pp['time_shift'],pp['time_max']*pp['time_dt']+pp['time_shift'],pp['time_nticks']))
	popplot.ax[0].set_yticks(popplot.figure_out_ticks(pp['fret_min'],pp['fret_max'],pp['fret_nticks']))

	fs = pp['label_fontsize']/dpr
	font = {
		'family': pp['font'],
		'size': fs,
		'va':'top'
	}
	popplot.ax[0].set_xlabel(pp['xlabel_text'],fontdict=font)
	popplot.ax[0].set_ylabel(pp['ylabel_text'],fontdict=font)

	popplot.ax[0].yaxis.set_label_coords(pp['ylabel_offset'], 0.5)
	popplot.ax[0].xaxis.set_label_coords(0.5, pp['xlabel_offset'])

	bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=1./dpr)
	lstr = 'n = %d'%(fpb.shape[0])
	if pp['textbox_nmol']:
		lstr = 'N = %d'%(fpb.shape[0])

	popplot.ax[0].annotate(lstr,xy=(pp['textbox_x'],pp['textbox_y']),xycoords='axes fraction',ha='right',color='k',bbox=bbox_props,fontsize=pp['textbox_fontsize']/dpr)

	popplot.f.canvas.draw()
