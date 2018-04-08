import numpy as np
import matplotlib.pyplot as plt

default_prefs = {
	'plotter_cm': 'Blues_r',
	'plotter_undercm': 'lightgray',
}

def plot(gui):
	popplot = gui.popout_plots['plot_tranM'].ui
	popplot.ax[0].cla()
	popplot.resize_fig()
	gui.app.processEvents()

	if not gui.data.hmm_result is None:
		if gui.data.hmm_result.type == 'consensus':
			A = gui.data.hmm_result.alpha
			m = gui.data.hmm_result.m
			for i2 in range(0,len(A)):
				A[i2,i2]=-1
			try:
				cm = plt.cm.__dict__[popplot.prefs['plotter_cm']]
			except:
				cm = plt.cm.rainbow
			try:
				cm.set_under(popplot.prefs['plotter_undercm'])
			except:
				cm.set_under('lightgray')
			im = popplot.ax[0].imshow(A.T, cmap=cm, origin='lower',interpolation='nearest', vmin=0)
			if len(popplot.f.axes) == 1:
				cb = popplot.f.colorbar(im)
			else:
				popplot.f.axes[1].cla()
				cb = popplot.f.colorbar(im,cax=popplot.f.axes[1])
			popplot.ax[0].set_xticks(np.arange(A.shape[0]))
			popplot.ax[0].set_xticklabels(["%.2f"%(mm) for mm in m],fontsize=popplot.prefs['label_ticksize'])
			popplot.ax[0].set_xlabel('Starting FRET State',fontsize=popplot.prefs['label_fontsize'])
			popplot.ax[0].set_yticks(np.arange(A.shape[1]))
			popplot.ax[0].set_yticklabels(["%.2f"%(mm) for mm in m],fontsize=popplot.prefs['label_ticksize'])
			popplot.ax[0].set_ylabel('Ending FRET State',fontsize=popplot.prefs['label_fontsize'])
			popplot.ax[0].set_title('Transition Count',fontsize=popplot.prefs['label_fontsize'])
			popplot.f.canvas.draw()
