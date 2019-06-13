import numpy as np
from PyQt5.QtCore import QTimer
import numba as nb

default_prefs = {
	'smooth_frames':.25
}


def region_select(eclick, erelease, gui):
	@nb.jit("float64[:](uint16[:,:,:],int32[:,:],int32,int32,int32,int32)")
	def average_region(d,bg,xmin,xmax,ymin,ymax):
		z = np.zeros(d.shape[0],dtype=nb.float64)
		n = float((xmax-xmin)*(ymax-ymin))
		if n == 0:
			n = 1
		for t in range(d.shape[0]):
			for i in range(xmin,xmax+1):
				for j in range(ymin,ymax+1):
					z[t] += float(d[t,i,j]-bg[i,j])/n
		return z
	# if not gui.popout_plots['plot_region'].isVisible():
	if 1:
		e = [int(ee) for ee in gui.plot.rectangle.extents]

		popplot = gui.popout_plots['plot_region'].ui
		popplot.ax[0].cla()#
		popplot.resize_fig()
		gui.app.processEvents()

		bg = gui.data.background
		if bg is None:
			bg = np.zeros(gui.data.movie.shape[1],gui.data.movie.shape[2],dtype='int32')
		y = average_region(gui.data.movie,bg.astype('int32'),e[2],e[3],e[0],e[1])
		from scipy.ndimage import gaussian_filter1d
		yy = gaussian_filter1d(y,popplot.prefs['smooth_frames'])
		popplot.ax[0].plot(yy,color='k',alpha=.9,lw=1)

		popplot.ax[0].set_xlim(0,gui.data.movie.shape[0])
		popplot.f.canvas.draw()

def plot(gui):


	# gui.plot.setup_rectangle()
	gui.plot.rectangle.onselect = lambda eclick,erelease: region_select(eclick,erelease,gui)

	# popcorn = gui.raise_plot(1)
	region_select(None,None,gui)
