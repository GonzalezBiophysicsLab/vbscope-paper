import numpy as np
import matplotlib.pyplot as plt

default_prefs = {
'subplots_hspace':.3,
'subplots_left':.15,
'subplots_top':.95,
'fig_height':5.0,
'xlabel_offset':-.15,
'ylabel_offset':-.15,

'time_scale':'log',
'time_dt':1.0,
'time_nticks':5,
'time_min':0.0,
'time_max':2000.0,

'acorr_nticks':6,
'acorr_min':-0.1,
'acorr_max':1.0,
'acorr_filter':1.0,
'acorr_highpass':0.,

'power_nticks':6,
'power_min':.1,
'power_max':100.0,

'line_color':'blue',
'line_linewidth':1,
'line_alpha':0.9,

'fill_alpha':0.3,
'fill_color':'blue',

'xlabel_rotate':0.,
'ylabel_rotate':0.,
'xlabel_text1':r'Time(s)',
'ylabel_text1':r'Autocorrelation Function',
'xlabel_text2':r'Frequency (s$^{-1}$)',
'ylabel_text2':r'Power Spectrum',
'xlabel_decimals':2,
'ylabel_decimals':2,

'textbox_x':0.95,
'textbox_y':0.93,
'textbox_fontsize':8
}



def plot(gui):
	popplot = gui.popout_plots['plot_acorr'].ui
	pp = popplot.prefs
	popplot.ax[0].cla()
	popplot.ax[1].cla()
	popplot.resize_fig()
	gui.app.processEvents()

	dpr = popplot.f.canvas.devicePixelRatio()

	if gui.ncolors != 2:
		return

	from ..supporting.autocorr import ensemble_bayes_acorr,credible_interval

	fpb = gui.data.get_plot_data(True)[0].copy() ## FRET from filtered intensities

	posterior = ensemble_bayes_acorr(fpb)
	ci = credible_interval(posterior)

	norm = posterior[0][0]
	t = pp['time_dt']*np.arange(posterior[0].size)

	yy = posterior[0]/norm
	from scipy.ndimage import gaussian_filter1d
	if pp['acorr_filter'] > 0.:
		yy = gaussian_filter1d(yy,pp['acorr_filter'])
		ci[0] = gaussian_filter1d(ci[0],pp['acorr_filter'])
		ci[1] = gaussian_filter1d(ci[1],pp['acorr_filter'])
	ct = np.sum(yy/yy[0])
	x = t < pp['acorr_highpass']
	yy[x] = 0.
	ci[:,x] = 0.

	popplot.ax[0].fill_between(t, ci[0]/norm, ci[1]/norm, alpha=.3, color=pp['line_color'])
	popplot.ax[0].plot(t, yy, color=pp['line_color'], lw=1., alpha=.9)
	popplot.ax[0].axvline(ct*pp['time_dt'], color='k', lw=1., alpha=.9)


	f = np.fft.fft(yy)
	ft = np.fft.fftfreq(yy.size) *(1./pp['time_dt'])
	x = ft.argsort()
	x = x[x>0]
	popplot.ax[1].semilogy(ft[x],np.abs(f)[x],lw=1.,color='b',alpha=.9)
	# popplot.ax[1].set_xlim(ft.min(),ft.max())
	# popplot.ax[1].set_ylim(0.01,100.)

	# # for i in range(fpb.shape[0]):
	# # 	y = acorr(fpb[i])
	# # 	popplot.ax[0].plot(t,y/y[0],alpha=.1,color='r')
	# y = posterior[0]
	# tt = pp['time_dt']*np.arange(y.size)
	# popplot.ax[0].fill_between(tt, ci[0]/y[0], ci[1]/y[0], alpha=pp['fill_alpha'], color=pp['fill_color'])
	# popplot.ax[0].plot(tt, y/y[0], color=pp['line_color'], lw=pp['line_linewidth'], alpha=pp['line_alpha'])
	# popplot.ax[0].plot(tt, q/q[0], color=pp['line_color'], lw=pp['line_linewidth'], alpha=pp['line_alpha'])

	# popplot.ax[0].set_ylim(-.1,1.)
	# popplot.ax[0].set_xlim(0,500.)
	# print y[0]
	# f = np.fft.fft(y/y[0])
	# ft = np.fft.fftfreq(y.size)
	# x = ft.argsort()
	# popplot.ax[0].plot(ft[x],np.abs(f)[x],lw=1.,color='b',alpha=.9)
	# # popplot.ax[0].plot(y/y[0],lw=1.,color='b',alpha=.9)
	#
	popplot.ax[0].set_xscale('linear')
	if pp['time_scale'] == 'log':
		popplot.ax[0].set_xscale('log')
		if pp['time_min'] < pp['time_dt']:
			pp['time_min'] = pp['time_dt']
	popplot.ax[0].set_xlim(pp['time_min'],pp['time_max'])
	popplot.ax[0].set_ylim(pp['acorr_min'],pp['acorr_max'])
	popplot.ax[1].set_ylim(pp['power_min'],pp['power_max'])

	popplot.ax[1].set_xlim(ft[ft>0].min(),ft[ft>0].max())
	popplot.ax[1].set_xscale('log')

	# ####################################################
	# ####################################################

	fs = pp['label_fontsize']/dpr
	font = {
		'family': pp['font'],
		'size': fs,
		'va':'top'
	}
	popplot.ax[0].set_xlabel(pp['xlabel_text1'],fontdict=font)
	popplot.ax[0].set_ylabel(pp['ylabel_text1'],fontdict=font)
	popplot.ax[0].yaxis.set_label_coords(pp['ylabel_offset'], 0.5)
	popplot.ax[0].xaxis.set_label_coords(0.5, pp['xlabel_offset'])
	popplot.ax[1].set_xlabel(pp['xlabel_text2'],fontdict=font)
	popplot.ax[1].set_ylabel(pp['ylabel_text2'],fontdict=font)
	popplot.ax[1].yaxis.set_label_coords(pp['ylabel_offset'], 0.5)
	popplot.ax[1].xaxis.set_label_coords(0.5, pp['xlabel_offset'])

	if not pp['time_scale'] == 'log':
		popplot.ax[0].set_xticks(popplot.figure_out_ticks(pp['time_min'],pp['time_max'],pp['time_nticks']))
	popplot.ax[0].set_yticks(popplot.figure_out_ticks(pp['acorr_min'],pp['acorr_max'],pp['acorr_nticks']))

	bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=1./dpr)
	lstr = 'N = %d'%(fpb.shape[0])
	popplot.ax[0].annotate(lstr,xy=(pp['textbox_x'],pp['textbox_y']),xycoords='axes fraction', ha='right', color='k', bbox=bbox_props, fontsize=pp['textbox_fontsize']/dpr)

	fd = {'rotation':pp['xlabel_rotate'], 'ha':'center'}
	if fd['rotation'] != 0: fd['ha'] = 'right'
	popplot.ax[0].set_xticklabels(["{0:.{1}f}".format(x, pp['xlabel_decimals']) for x in popplot.ax[0].get_xticks()], fontdict=fd)

	fd = {'rotation':pp['ylabel_rotate']}
	popplot.ax[0].set_yticklabels(["{0:.{1}f}".format(y, pp['ylabel_decimals']) for y in popplot.ax[0].get_yticks()], fontdict=fd)

	popplot.f.canvas.draw()
