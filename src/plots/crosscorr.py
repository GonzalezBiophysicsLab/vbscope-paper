import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.signal import wiener

default_prefs = {
	# 'label_x_nticks':7,
	# 'label_y_nticks':7,
	'label_ticksize':8,
	'label_padding':.17,
	'wiener_filter':False

}



def get_data(gui):
	d = gui.data.d.copy()
	if gui.popout_plots['crosscorr'].ui.prefs['wiener_filter'] is True:
		for i in range(d.shape[0]):
			for j in range(d.shape[1]):
				try:
					d[i,j] = wiener(d[i,j])
				except:
					pass
	q = np.array([gui.data.calc_cross_corr(d[i,:,gui.data.pre_list[i]:gui.data.pb_list[i]]) for i in range(d.shape[0])])
	checked = gui.classes_get_checked()
	n = np.arange(q.size)
	return n[checked],q[checked]

def plot(gui):
	popplot = gui.popout_plots['crosscorr'].ui
	popplot.ax[0].cla()
	popplot.resize_fig()
	gui.app.processEvents()

	if gui.ncolors == 2:
		n,q = get_data(gui)

		popplot.ax[0].plot(n,q,'o',color='k')
		popplot.ax[0].set_yscale('symlog')
		popplot.ax[0].set_xlim(0,n.size)


		popplot.ax[0].set_xlabel('Trajectory',fontsize=popplot.prefs['label_fontsize']/gui.plot.canvas.devicePixelRatio())
		popplot.ax[0].set_ylabel('Cross Correlation',fontsize=popplot.prefs['label_fontsize']/gui.plot.canvas.devicePixelRatio())

		popplot.ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
		popplot.ax[0].set_xlim(-1,n[-1]+1)
		# popplot.ax[0].set_xticks(np.linspace(0,n.size,popplot.prefs['label_x_nticks']))



		qmin = np.sign(q.min())*10.**(np.floor(np.log10(np.abs(q.min()))))
		qmax = np.sign(q.max())*10.**(np.ceil(np.log10(np.abs(q.max()))))
		if qmin == qmax:
			qmin /= 10.
		if qmin > qmax:
			tmp = qmin
			qmin = qmax
			qmax = tmp

		popplot.ax[0].set_ylim(qmin,qmax)

		yticks = popplot.ax[0].get_yticks()
		popplot.ax[0].set_yticklabels(["%.1e"%(tt) for tt in yticks])

		for asp in ['top','bottom','left','right']:
			popplot.ax[0].spines[asp].set_linewidth(1.0/gui.plot.canvas.devicePixelRatio())
		popplot.f.subplots_adjust(left=.05+popplot.prefs['label_padding'],bottom=.05+popplot.prefs['label_padding'],top=.95,right=.95)



		popplot.f.canvas.draw()
