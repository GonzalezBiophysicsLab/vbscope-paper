from PyQt5.QtWidgets import QWidget, QSizePolicy,QGridLayout,QLabel,QSpinBox,QMessageBox, QFileDialog, QPushButton, QInputDialog
from PyQt5.QtCore import Qt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class mpl_plot():
	def __init__(self):
		self.f,self.ax = plt.subplots(1,figsize=(4,3))
		self.canvas = FigureCanvas(self.f)
		self.toolbar = NavigationToolbar(self.canvas,None)

		sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
		self.canvas.setSizePolicy(sizePolicy)
		self.f.subplots_adjust(left=.08,right=.92,top=.92,bottom=.08)

		self.ax.tick_params(axis='both', which='major', labelsize=8)
		self.canvas.draw()
		plt.close(self.f)

class dock_mesoscopic(QWidget):
	def __init__(self,parent=None):
		super(dock_mesoscopic, self).__init__(parent)

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
		ny, success1 = QInputDialog.getInt(self,"Number X","Number of pixels (X)",value=50,min=1)
		nx, success2 = QInputDialog.getInt(self,"Number Y","Number of pixels (Y)",value=50,min=1)
		if not success1 or not success2:
			return None

		## Define the start and end frames
		start, success1 = QInputDialog.getInt(self,"Start Frame","Start Frame",value=0,min=0,max=self.gui.data.total_frames)
		end,   success2 = QInputDialog.getInt(self,"Start Frame","Number of pixels (Y)",value=self.gui.data.total_frames,min=start+1,max=self.gui.data.total_frames)
		if not success1 or not success2:
			return

		rects = []
		regions,shifts = self.gui.data.regions_shifts()
		c = self.gui.prefs['channel_colors']

		t = np.arange(start,end)
		data = np.zeros((self.gui.data.ncolors,end-start))

		for i in range(self.gui.data.ncolors):
			region = regions[i]
			ox = (region[0,1] + region[0,0])/2
			oy = (region[1,1] + region[1,0])/2

			rect = Rectangle((oy-ny/2,ox-nx/2), ny, nx, ec=c[i], fill=False, alpha=.95, lw=1.5)
			rects.append(rect)

			movie = self.gui.data.movie[start:end,ox-nx/2:ox+nx/2,oy-ny/2:oy+ny/2]
			data[i] = acorr(movie)#movie.mean((1,2))
		pc = PatchCollection(rects,match_original=True)

		self.gui.plot.clear_collections()
		self.gui.plot.ax.add_collection(pc)
		self.gui.plot.canvas.draw()

		print data.shape,self.gui.data.movie.shape
		self.plot.ax.cla()
		for i in range(self.gui.data.ncolors):
			self.plot.ax.plot(t,data[i],color=c[i],lw=1.,alpha=.95)
		self.plot.ax.set_xlim(t.min(),t.max())
		self.plot.canvas.draw()

# @jit("double[:,:](double[:,:,:])")
# @jit("double[:](double[:,:,:])")

try:
	nb.cuda.detect()
	_flag_cuda = True
except:
	_flag_cuda = False

if _flag_cuda:
	@nb.cuda.jit(argtypes=(nb.int16[:,:,:],nb.float32[:]))
	def _kernel(movie,y):
		x,y = cuda.grid(2)
		if x < movie.shape[1] and y < movie.shape[2]:
			for i in range(movie.shape[0]):
				for j in range(i,movie.shape[0]):
					y[j-i] += float(movie[i,x,y])*float(movie[j,x,y]) / float(movie.shape[0] - (j-i)) / float(movie.shape[1]*movie.shape[2])
	def acorr(x):
		div = 64
		grids = (x.shape[1]/div,y.shape[2]/div)
		threads = (div,div)

		print nb.cuda.get_current_device()
		y = np.zeros(x.shape[0],dtype='float32')
		_kernel[grids,threads](x,y)
		cuda.synchronize()
		return y

else:
	@nb.jit("double[:](double[:,:,:])")
	def acorr(x):
		y = np.zeros(x.shape[0])
		for i in range(y.size):
			for j in range(i,y.size):
				y[j-i] += x[i,0,0]*x[j,0,0] / (y.size - (j-i))
		return y

		# n = x.shape[1]*x.shape[2]
		# y = np.zeros_like(x)
		# for ix in range(y.shape[1]):
		# 	for iy in range(y.shape[2]):
		# 		print ix*x.shape[1] + iy
		# 		for i in range(y.shape[0]):
		# 			for j in range(i,y.shape[0]):
		# 				y[j-i,ix,iy] += x[i,ix,iy]*x[j,ix,iy] / (y.shape[0] - (j-i)) / n
		#
		# return np.sum(np.sum(y,1),1)
		# #
		# # Priors
		# a0 = .5
		# k0 = 1.
		# m0 = y.mean()
		# b0 = 1.
		#
		# n = np.arange(y.size,dtype='f')[::-1] + 1.
		#
		# # Posteriors
		# an = a0 + n/2.
		# kn = k0 + n
		# mn = (k0*m0 + n*y)/(kn)
		# bn = b0 + k0*n*(y-m0)**2. / (2.*kn)
		#
		# for i in range(y.size):
		# 	for j in range(i,y.size):
		# 		bn[j-i] += .5*(x[i]*x[j] - y[j-i])**2.
		#
		# return np.array([mn,kn,an,bn])
