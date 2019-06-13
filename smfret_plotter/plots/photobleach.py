import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QPushButton

class obj(object): ## generic class to take anything you throw at it...
	def __init__(self,*args):
		self.args = args

default_prefs = {
	'fig_height':2.0,
	'fig_width':3.5,
	'subplots_top':0.95,
	'subplots_left':0.15,
	'subplots_bottom':0.22,
	'xlabel_offset':-0.13,

	'surv_log_y':True,
	'surv_normalize':True,
	'surv_nticks':5,
	'surv_line_half':False,
	'surv_line_ln':False,
	'surv_fit':True,
	'surv_fit_type':1,
	# 'surv_line_annotate':True,
	# 'surv_annotate_dec':2,

	'time_dt':1.0,
	'time_nticks':5,

	'textbox_x':0.965,
	'textbox_y':0.9,
	'textbox_fontsize':8.0,
	'textbox_nmol':True,

	'xlabel_text':r'Time',
	'ylabel_text':r'Survival',

}

def setup(gui):
	gui.popout_plots['plot_photobleach'].ui.pb = obj()

	if not gui.data.d is None:
		# recalc(gui)
		plot(gui)

# def recalc(gui):
# 	## Data
# 	t = np.arange(gui.data.d.shape[2])
# 	pre = gui.data.pre_list
# 	post = gui.data.post_list
# 	checked = gui.classes_get_checked()
#
# 	pb = (post-pre)[checked]
#
# 	survival = np.sum(pb[:,None] > t[None,:],axis=0)
#
# 	gui.popout_plots['plot_photobleach'].ui.pb.t = t
# 	gui.popout_plots['plot_photobleach'].ui.pb.survival = survival
# 	gui.popout_plots['plot_photobleach'].ui.pb.n = checked.sum()
#
# 	plot(gui)


def plot(gui):
	if gui.data.d is None:
		return

	t = np.arange(gui.data.d.shape[2])
	pre = gui.data.pre_list
	post = gui.data.post_list
	checked = gui.classes_get_checked()

	# pb = (post-pre)[checked]
	pb = post[checked] ##

	survival = np.sum(pb[:,None] > t[None,:],axis=0)

	gui.popout_plots['plot_photobleach'].ui.pb.t = t
	gui.popout_plots['plot_photobleach'].ui.pb.survival = survival
	gui.popout_plots['plot_photobleach'].ui.pb.n = checked.sum()

	popplot = gui.popout_plots['plot_photobleach'].ui
	pp = popplot.prefs
	popplot.ax[0].cla()
	popplot.resize_fig()
	gui.app.processEvents()

	t = popplot.pb.t * pp['time_dt']
	s = popplot.pb.survival
	if pp['surv_normalize']:
		s = s/s[0]

	tt = np.zeros((2*t.size))
	ss = np.zeros((2*t.size))

	tt[::2] = t
	tt[1::2] = t
	ss[::2] = s
	ss[1::2] = np.roll(s,-1)

	tt = tt[:-1]
	ss = ss[:-1]

	if pp['surv_line_half']:
		ind = (s<s[0]/2.).argmax()
		gui.log('photobleaching half: t = %f'%(t[ind]))
		popplot.ax[0].plot([t[ind],t[ind]],[0.,s[ind]],color='r',alpha=.8)
		popplot.ax[0].plot([t[0],t[ind]],[s[ind],s[ind]],color='r',alpha=.8)
		# if pp['surv_line_annotate']:
		# 	popplot.ax[0].annotate('t={:.{precision}f}'.format(t[ind],precision=pp['surv_annotate_dec']), xy=(t[ind], s[ind]), xycoords='data', xytext=(t[ind]+(t[-1]-t[0])/20.,0.55), textcoords='axes fraction', arrowprops=dict(facecolor='black', shrink=0.05))

	if pp['surv_line_ln']:
		ind = (s<s[0]*np.exp(-1.)).argmax()
		gui.log('photobleaching ln: t = %f'%(t[ind]))
		popplot.ax[0].plot([t[ind],t[ind]],[0.,s[ind]],color='r',alpha=.8)
		popplot.ax[0].plot([t[0],t[ind]],[s[ind],s[ind]],color='r',alpha=.8)
		# if pp['surv_line_annotate']:
		# 	popplot.ax[0].annotate('t={:.{precision}f}'.format(t[ind],precision=pp['surv_annotate_dec']), xy=(t[ind], s[ind]), xycoords='data', xytext=(t[ind]+(t[-1]-t[0])/20.,0.55), textcoords='axes fraction', arrowprops=dict(facecolor='black', shrink=0.05))



	popplot.ax[0].plot(tt,ss,color='k',lw=1,alpha=.8)

	if pp['surv_fit']:
		from ..supporting.autocorr import fit_exponential,fit_stretched#,fit_biexponential
		y = s.copy()/s[0]
		yy = y<0.05
		if np.any(yy > 0):
			cutoff = np.argmax(yy)
			if cutoff > y.size:
				cutoff = -1
		else:
			cutoff = -1
		start = 0

		if pp['surv_fit_type'] == 1:
			fit = fit_exponential(t[start:cutoff],y[start:cutoff])
			z = fit(t)*s[0]
		# elif pp['surv_fit_type'] == 2:
			# fit = fit_biexponential(t[start:cutoff],y[start:cutoff])
			# z = fit(t)*s[0]
		elif pp['surv_fit_type'] == 0:
			fit = fit_stretched(t[start:cutoff],y[start:cutoff])
			z = fit(t)*s[0]
		else:
			fit = None

		if not fit is None:
			popplot.ax[0].plot(t,z,color='r',alpha=.8,lw=1)
			gui.log('photobleaching fit: %s, %s'%(fit.type,str(fit.params)))
			gui.log('photobleaching t_c: %f'%(fit.calc_tc()))
	popplot.f.canvas.draw()

	ylim = popplot.ax[0].get_ylim()
	smin = 0.
	smax = ss.max()
	if pp['surv_log_y']:
		smin = (ss[ss>0].min())
		smin -= smin**0.1
		popplot.ax[0].set_yscale('log')

	delta = smax-smin
	if not pp['surv_log_y']:
		smax += delta*.05
		ticks = popplot.figure_out_ticks(smin,smax,pp['surv_nticks'])
		popplot.ax[0].set_yticks(ticks)
	popplot.ax[0].set_ylim(smin,smax)

	ticks = popplot.figure_out_ticks(t[0],t[-1],pp['time_nticks'])
	popplot.ax[0].set_xticks(ticks)
	popplot.ax[0].set_xlim(t[0],t[-1])

	dpr = popplot.f.canvas.devicePixelRatio()

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
	lstr = 'N = %d'%(popplot.pb.n)

	popplot.ax[0].annotate(lstr,xy=(pp['textbox_x'],pp['textbox_y']),xycoords='axes fraction',ha='right',color='k',bbox=bbox_props,fontsize=pp['textbox_fontsize']/dpr,family=pp['font'])

	popplot.f.canvas.draw()
