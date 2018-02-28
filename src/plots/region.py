import numpy as np
from PyQt5.QtCore import QTimer
import numba as nb

@nb.jit("float64[:](uint16[:,:,:],int32,int32,int32,int32)")
def average_region(d,xmin,xmax,ymin,ymax):
	z = np.zeros(d.shape[0],dtype=nb.float64)
	n = float((xmax-xmin)*(ymax-ymin))
	if n == 0:
		n = 1
	for t in range(d.shape[0]):
		for i in range(xmin,xmax+1):
			for j in range(ymin,ymax+1):
				z[t] += float(d[t,i,j])/n
	return z


def region_select(eclick, erelease, gui):
	if not gui.docks['pc_region'][0].isHidden():
		e = [int(ee) for ee in gui.plot.rectangle.extents]

		print gui.data.movie.shape
		print gui.plot.rectangle.center
		print gui.plot.rectangle.extents
		popcorn = gui.docks['pc_region'][1]
		popcorn.ax[0].cla()

		y = average_region(gui.data.movie,e[2],e[3],e[0],e[1])
		from scipy.ndimage import gaussian_filter1d
		yy = gaussian_filter1d(y,.25)
		popcorn.ax[0].plot(yy,color='k',alpha=.9,lw=1)

		popcorn.ax[0].set_xlim(0,gui.data.movie.shape[0])
		popcorn.f.canvas.draw()

def region(gui):


	# gui.plot.setup_rectangle()
	gui.plot.rectangle.onselect = lambda eclick,erelease: region_select(eclick,erelease,gui)

	# popcorn = gui.raise_plot(1)
	region_select(None,None,gui)
