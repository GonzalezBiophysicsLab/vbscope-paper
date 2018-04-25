import numpy as np
import matplotlib.pyplot as plt

default_prefs = {
	'fig_height':4.0,
	'fig_width':4.0,
	'label_padding_left':.20,
	'label_padding_bottom':.15,
	'label_padding_top':.05,
	'label_padding_right':.05,
	'label_space':5.,

	'bar_color':'steelblue',
	'bar_edgecolor':'black',

	'states_low':1,
	'states_high':10,

	'label_y_nticks':4,

	'textbox_x':0.96,
	'textbox_y':0.93,
	'textbox_fontsize':10.0,
	'textbox_nmol':True
}


def plot(gui):
	popplot = gui.popout_plots['vb_states'].ui
	popplot.ax[0].cla()
	popplot.resize_fig()
	gui.app.processEvents()

	if not gui.data.hmm_result is None:
		# if gui.data.hmm_result.__dict__.keys().count('models')
		if gui.data.hmm_result.type == 'vb':
			ns = np.arange(popplot.prefs['states_low'],popplot.prefs['states_high']+1).astype('i')
			nstates = np.array([r.mu.size for r in gui.data.hmm_result.results])
			y = np.array([np.sum(nstates == i) for i in ns])


			popplot.ax[0].bar(ns,y,width=1.0,color=popplot.prefs['bar_color'],edgecolor=popplot.prefs['bar_edgecolor'])

		popplot.ax[0].set_xticks(ns)

		popplot.ax[0].set_xlabel(r'$N_{\rm states}$',fontsize=popplot.prefs['label_fontsize']/gui.plot.canvas.devicePixelRatio(),labelpad=popplot.prefs['label_space'])
		popplot.ax[0].set_ylabel(r'$N_{\rm trajectories}$', fontsize=popplot.prefs['label_fontsize']/gui.plot.canvas.devicePixelRatio(), labelpad=popplot.prefs['label_space'])

		for asp in ['top','bottom','left','right']:
			popplot.ax[0].spines[asp].set_linewidth(1.0/gui.plot.canvas.devicePixelRatio())
		popplot.f.subplots_adjust(left=popplot.prefs['label_padding_left'],bottom=popplot.prefs['label_padding_bottom'],top=1.-popplot.prefs['label_padding_top'],right=1.-popplot.prefs['label_padding_right'])

		bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=1./gui.plot.canvas.devicePixelRatio())
		lstr = 'N = %d'%(int(y.sum()))

		popplot.ax[0].annotate(lstr,xy=(popplot.prefs['textbox_x'],popplot.prefs['textbox_y']),xycoords='axes fraction',ha='right',color='k',bbox=bbox_props,fontsize=popplot.prefs['textbox_fontsize']/gui.plot.canvas.devicePixelRatio())

		popplot.f.canvas.draw()
