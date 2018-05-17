import numpy as np
import matplotlib.pyplot as plt

default_prefs = {
'fret_min':-.2,
'fret_max':1.2,
'fret_nbins':41,
'hist_smoothx':4.,
'hist_smoothy':4.,

'hist_inerp_res':800,

'color_cmap':'rainbow',
'color_floorcolor':'lightgoldenrodyellow',
'color_ceiling':0.0,
'color_floor':0.0,
'color_nticks':5,
'hist_rawsignal':True,
'hist_normalize':True,

'label_y_nticks':8,
'label_x_nticks':8,
'label_ticksize':8,
'label_x_rotate',:0.,
'label_y_rotate',:0.,

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
	popplot.ax[0].cla()
	popplot.resize_fig()
	gui.app.processEvents()

	if gui.ncolors != 2:
		return

	d1,d2,N = get_neighbor_data(gui)
	x,y,z = gen_histogram(gui,d1,d2)

	### Plotting
	cm = colormap(gui)

	if popplot.prefs['color_ceiling'] == popplot.prefs['color_floor']:
		popplot.prefs['color_floor'] = 0.0
		popplot.prefs['color_ceiling'] = np.ceil(z.max())
		popplot._prefs.update_table()

	vmin = popplot.prefs['color_floor']
	vmax = popplot.prefs['color_ceiling']

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

	popplot.ax[0].set_xlabel(r'Initial E$_{\rm FRET}$',fontsize=popplot.prefs['label_fontsize']/gui.plot.canvas.devicePixelRatio())
	popplot.ax[0].set_ylabel(r'Final E$_{\rm FRET}$',fontsize=popplot.prefs['label_fontsize']/gui.plot.canvas.devicePixelRatio())
	popplot.ax[0].set_title('Transition Density',fontsize=popplot.prefs['label_fontsize']/gui.plot.canvas.devicePixelRatio())
	# bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=1./gui.plot.canvas.devicePixelRatio())
	# popplot.ax[0].annotate('n = %d'%(fpb.shape[0]),xy=(.95,.93),xycoords='axes fraction',ha='right',color='k',bbox=bbox_props,fontsize=popplot.prefs['label_ticksize']/gui.plot.canvas.devicePixelRatio())
	popplot.ax[0].set_xticks(np.around(np.linspace(popplot.prefs['fret_min'],popplot.prefs['fret_max'],popplot.prefs['label_x_nticks']),4)+0.)
	popplot.ax[0].set_xticklabels(popplot.ax[0].get_xticks(),rotation=popplot.prefs['label_x_rotate'])
	popplot.ax[0].set_yticks(np.around(np.linspace(popplot.prefs['fret_min'],popplot.prefs['fret_max'],popplot.prefs['label_y_nticks']),4)+0.)
	popplot.ax[0].set_yticklabels(popplot.ax[0].get_yticks(),rotation=popplot.prefs['label_y_rotate'])


	bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=1./gui.plot.canvas.devicePixelRatio())
	lstr = 'n = %d'%(d1.size)
	if popplot.prefs['textbox_nmol']:
		lstr = 'N = %d'%(N)

	popplot.ax[0].annotate(lstr,xy=(popplot.prefs['textbox_x'],popplot.prefs['textbox_y']),xycoords='axes fraction',ha='right',color='k',bbox=bbox_props,fontsize=popplot.prefs['textbox_fontsize']/gui.plot.canvas.devicePixelRatio())


	for asp in ['top','bottom','left','right']:
		popplot.ax[0].spines[asp].set_linewidth(1.0/gui.plot.canvas.devicePixelRatio())
	popplot.f.subplots_adjust(left=.05+popplot.prefs['label_padding'],bottom=.05+popplot.prefs['label_padding'],top=.95-popplot.prefs['label_padding'],right=.95)

	popplot.f.canvas.draw()
