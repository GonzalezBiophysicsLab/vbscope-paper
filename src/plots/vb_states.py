import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import cnames

default_prefs = {
	'bar_color':'steelblue',
	'bar_edgecolor':'black',

	'states_low':1,
	'states_high':10,

	'count_nticks':5,


	'xlabel_text':r'N$_{\rm{States}}$',
	'ylabel_nticks':4,
	'ylabel_text':r'$N_{\rm trajectories}$',

	'textbox_x':0.96,
	'textbox_y':0.93,
	'textbox_fontsize':8.0,
	'textbox_nmol':True,

	'fig_width':2.0,
	'fig_height':2.0,
	'subplots_top':0.97,
	'subplots_left':0.2
}

def plot(gui):
	popplot = gui.popout_plots['vb_states'].ui
	pp = popplot.prefs
	popplot.ax[0].cla()
	popplot.resize_fig()
	gui.app.processEvents()

	dpr = popplot.f.canvas.devicePixelRatio()

	if not gui.data.hmm_result is None:
		if gui.data.hmm_result.type == 'vb':
			ns = np.arange(pp['states_low'],pp['states_high']+1).astype('i')
			nstates = np.array([r.mu.size for r in gui.data.hmm_result.results])
			y = np.array([np.sum(nstates == i) for i in ns])

			try:
				bcolor = 'steelblue'
				ecolor = 'black'
				if list(cnames.keys()).count(pp['bar_color']) > 0:
					bcolor = pp['bar_color']
				if list(cnames.keys()).count(pp['bar_edgecolor']) > 0:
					ecolor = pp['bar_edgecolor']
				popplot.ax[0].bar(ns,y,width=1.0,color=bcolor,edgecolor=ecolor)
			except:
				pass

		popplot.ax[0].set_xticks(ns)
		ylim = popplot.ax[0].get_ylim()
		ticks = popplot.figure_out_ticks(0.,ylim[1],pp['count_nticks'])
		popplot.ax[0].set_yticks(ticks)

		popplot.ax[0].set_xlabel(pp['xlabel_text'],fontsize=pp['label_fontsize']/dpr,labelpad=popplot.prefs['xlabel_offset'])
		popplot.ax[0].set_ylabel(pp['ylabel_text'],fontsize=pp['label_fontsize']/dpr,labelpad=popplot.prefs['ylabel_offset'])


		bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=pp['axes_linewidth']/dpr)
		lstr = 'N = %d'%(int(y.sum()))

		popplot.ax[0].annotate(lstr,xy=(pp['textbox_x'],pp['textbox_y']),xycoords='axes fraction',ha='right',color='k',bbox=bbox_props,fontsize=pp['textbox_fontsize']/dpr)

		popplot.f.canvas.draw()
