from PyQt5.QtWidgets import QWidget, QSizePolicy,QGridLayout,QLabel,QSpinBox,QMessageBox, QFileDialog, QPushButton, QInputDialog
from PyQt5.QtCore import Qt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# class mpl_plot():
# 	def __init__(self):
# 		self.f,self.ax = plt.subplots(1,figsize=(4,3))
# 		self.ax = [self.ax]
# 		self.canvas = FigureCanvas(self.f)
# 		self.toolbar = NavigationToolbar(self.canvas,None)
#
# 		sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
# 		self.canvas.setSizePolicy(sizePolicy)
# 		self.f.subplots_adjust(left=.08,right=.92,top=.92,bottom=.08)
#
# 		[aa.tick_params(axis='both', which='major', labelsize=8) for aa in self.ax]
# 		for aa in self.ax:
# 			aa.format_coord = lambda x, y: ''
# 		self.canvas.draw()
# 		self.f.tight_layout()
# 		plt.close(self.f)


from ..ui import gui
from ..containers import popout_plot_container

default_prefs = {
}

class dock_mesoscopic(QWidget):
	def __init__(self,parent=None):
		super(dock_mesoscopic, self).__init__(parent)

		self.default_prefs = default_prefs

		self.gui = parent

		self.layout = QGridLayout()

		self.button_calc = QPushButton("Calculate")
		self.button_calc.clicked.connect(self.calculate)

		self.plot = mpl_plot()
		self.layout.addWidget(self.plot.canvas,0,0)
		self.layout.addWidget(self.plot.toolbar,1,0)
		self.layout.addWidget(self.button_calc,2,0)

		self.setLayout(self.layout)

	def calculate(self):
		## Make sure a movie is loaded
		if not self.gui.data.flag_movie:
			return None

		## Define the region area
		ny, success1 = QInputDialog.getInt(self,"Number X","Number of pixels (X)",value=100,min=1)
		nx, success2 = QInputDialog.getInt(self,"Number Y","Number of pixels (Y)",value=100,min=1)
		if not success1 or not success2:
			return None

		## Define the start and end frames
		start, success1 = QInputDialog.getInt(self,"Start Frame","Start Frame",value=0,min=0,max=self.gui.data.total_frames)
		end,   success2 = QInputDialog.getInt(self,"Start Frame","Number of pixels (Y)",value=self.gui.data.total_frames,min=start+10,max=self.gui.data.total_frames)
		if not success1 or not success2:
			return

		rects = []
		regions,shifts = self.gui.data.regions_shifts()
		c = self.gui.prefs['channels_colors']

		t = np.arange(start,end)

		[aa.cla() for aa in self.plot.ax]
		m = []
		for i in range(self.gui.data.ncolors):
			region = regions[i]
			ox = (region[0,1] + region[0,0])/2
			oy = (region[1,1] + region[1,0])/2

			rect = Rectangle((oy-ny/2,ox-nx/2), ny, nx, ec=c[i], fill=False, alpha=.95, lw=1.5)
			rects.append(rect)

			movie = np.copy(self.gui.data.movie[start:end,ox-nx/2:ox+nx/2,oy-ny/2:oy+ny/2],order='C').astype('double')
			m.append(movie.mean((1,2)))
			self.plot.ax[0].plot(m[i],color=c[i],lw=1,alpha=.4)
			out = fit(t,m[i])
			rvs = np.random.multivariate_normal(out[0],out[1],size=1000)
			yrvs = np.array([powerlaw(t,*rvs[j]) for j in range(rvs.shape[0])])
			lb = np.percentile(yrvs,2.5,axis=0)
			ub = np.percentile(yrvs,97.5,axis=0)
			self.plot.ax[0].fill_between(t,lb,ub,color=c[i],alpha=.4)
			self.plot.ax[0].plot(t,powerlaw(t,*out[0]),color=c[i])
			# self.plot.ax[0].plot(t,ub,color=c[i],ls='-')
			print('color %d:'%(i))
			print('	k:',out[0][2])
			print('	n:',out[0][3])
			print('	<t>: %f +/- %f'%(e_t_powerlaw(out[0][2],out[0][3]),np.std(np.array([e_t_powerlaw(*rvs[j,2:]) for j in range(2)]))))

		pc = PatchCollection(rects,match_original=True)

		self.gui.plot.clear_collections()
		self.gui.plot.ax.add_collection(pc)
		self.gui.plot.canvas.draw()

		self.plot.ax[0].set_xlim(t.min(),t.max())
		self.plot.canvas.draw()

		oname = QFileDialog.getSaveFileName(self, 'Save Decays', '_meso.dat','*.dat')
		if oname[0] != "":
			np.savetxt(oname[0],np.array(m))

def powerlaw(t,a,b,k,n):
	return a*np.exp(-(k*t)**n) + b
def p_powerlaw(t,k,n):
	from scipy.special import gamma
	return k*gamma(1.+1./n)*np.exp(-(k*t)**n)
def e_t_powerlaw(k,n):
	from scipy.special import gamma
	return gamma((2.+n)/n)/gamma(1.+1./n)/2./k
def fit(t,y):
	from scipy.optimize import curve_fit
	p,c = curve_fit(powerlaw,t,y,p0=(y[0]-y[-1],y[-1],1./(t.size/2.),1.),maxfev=1000000)
	return p,c
