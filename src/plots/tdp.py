import numpy as np
import matplotlib.pyplot as plt

default_prefs = {
'subplots_top':0.97,
'subplots_right':0.91,
'subplots_left':0.05,
'axes_topright':True,
'xlabel_offset':-0.1,
'ylabel_offset':-0.18,

'fret_min':-.25,
'fret_max':1.25,
'fret_nbins':51,
'fret_nticks':7,

'hist_smoothx':1.,
'hist_smoothy':1.,
'hist_inerp_res':800,
'hist_rawsignal':True,
'hist_normalize':True,

'color_cmap':'rainbow',
'color_floorcolor':'white',
'color_ceiling':.95,
'color_floor':0.02,
'color_nticks':5,

'xlabel_rotate':0.,
'ylabel_rotate':0.,
'xlabel_text':r'Initial E$_{\rm{FRET}}$',
'ylabel_text':r'Final E$_{\rm{FRET}}$',
'xlabel_decimals':2,
'ylabel_decimals':2,

'textbox_x':0.95,
'textbox_y':0.93,
'textbox_fontsize':8,
'textbox_nmol':True
}


def get_neighbor_data(gui):
	fpb = gui.data.get_plot_data()[0]
	N = fpb.shape[0]
	d = np.array([[fpb[i,:-1],fpb[i,1:]] for i in range(fpb.shape[0])])

	if not gui.data.hmm_result is None:
		v = gui.data.get_viterbi_data(signal=True)
		vv = np.array([[v[i,:-1],v[i,1:]] for i in range(v.shape[0])])

		for i in range(d.shape[0]):
			d[i,:,vv[i,0]==vv[i,1]] = np.array((np.nan,np.nan))
			if not gui.popout_plots['plot_tdp'].ui.prefs['hist_rawsignal']:
				xx = np.nonzero(vv[i,0]!=vv[i,1])[0]
				d[i,0,xx] = v[i,xx]
				d[i,1,xx] = v[i,xx+1]

	d1 = d[:,0].flatten()
	d2 = d[:,1].flatten()
	cut = np.isfinite(d1)*np.isfinite(d2)

	return d1[cut],d2[cut],N

def gen_histogram(gui,d1,d2):
	from scipy.ndimage import gaussian_filter
	from scipy.interpolate import interp2d

	p = gui.popout_plots['plot_tdp'].ui.prefs

	## make histogram
	x = np.linspace(p['fret_min'],p['fret_max'],p['fret_nbins'])
	z,hx,hy = np.histogram2d(d1,d2,bins=[x.size,x.size],range=[[x.min(),x.max()],[x.min(),x.max()]])

	## smooth histogram
	z = gaussian_filter(z,(p['hist_smoothx'],p['hist_smoothy']))

	## interpolate histogram
	f =  interp2d(x,x,z, kind='cubic')
	x = np.linspace(p['fret_min'],p['fret_max'],p['hist_inerp_res'])
	z = f(x,x)
	z[z<0] = 0.

	if p['hist_normalize']:
		z /= z.max()

	return x,x,z

def colormap(gui):
	prefs = gui.popout_plots['plot_tdp'].ui.prefs

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
	popplot = gui.popout_plots['plot_tdp'].ui
	pp = popplot.prefs
	popplot.ax[0].cla()
	popplot.resize_fig()
	gui.app.processEvents()

	dpr = popplot.f.canvas.devicePixelRatio()

	if gui.ncolors != 2:
		return

	d1,d2,N = get_neighbor_data(gui)
	x,y,z = gen_histogram(gui,d1,d2)

	### Plotting
	cm = colormap(gui)

	if pp['color_ceiling'] == pp['color_floor']:
		pp['color_floor'] = 0.0
		pp['color_ceiling'] = np.ceil(z.max())

	vmin = pp['color_floor']
	vmax = pp['color_ceiling']

	pc = popplot.ax[0].imshow(z, cmap=cm, origin='lower',interpolation='none',extent=[x.min(),x.max(),x.min(),x.max()],vmin=vmin,vmax=vmax)

	# for pcc in pc.collections:
		# pcc.set_edgecolor("face")

	### Colorbar
	ext='neither'
	if vmin > z.min():
		ext = 'min'
	if len(popplot.f.axes) == 1:
		cb = popplot.f.colorbar(pc,extend=ext)
	else:
		popplot.f.axes[1].cla()
		cb = popplot.f.colorbar(pc,cax=popplot.f.axes[1],extend=ext)

	cbticks = np.linspace(vmin,vmax,pp['color_nticks'])
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

	popplot.ax[0].set_xticks(popplot.figure_out_ticks(pp['fret_min'],pp['fret_max'],pp['fret_nticks']))
	popplot.ax[0].set_yticks(popplot.figure_out_ticks(pp['fret_min'],pp['fret_max'],pp['fret_nticks']))

	bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=1./dpr)
	lstr = 'n = %d'%(d1.size)
	if pp['textbox_nmol']:
		lstr = 'N = %d'%(N)

	popplot.ax[0].annotate(lstr,xy=(pp['textbox_x'],pp['textbox_y']),xycoords='axes fraction', ha='right', color='k', bbox=bbox_props, fontsize=pp['textbox_fontsize']/dpr)

	fd = {'rotation':pp['xlabel_rotate'], 'ha':'center'}
	if fd['rotation'] != 0: fd['ha'] = 'right'
	popplot.ax[0].set_xticklabels(["{0:.{1}f}".format(x, pp['xlabel_decimals']) for x in popplot.ax[0].get_xticks()], fontdict=fd)

	fd = {'rotation':pp['ylabel_rotate']}
	popplot.ax[0].set_yticklabels(["{0:.{1}f}".format(y, pp['ylabel_decimals']) for y in popplot.ax[0].get_yticks()], fontdict=fd)

	popplot.f.canvas.draw()
