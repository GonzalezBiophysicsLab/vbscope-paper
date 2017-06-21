from PyQt5.QtWidgets import QMainWindow, QWidget, QSizePolicy, QVBoxLayout, QShortcut, QSlider, QHBoxLayout, QPushButton, QFileDialog, QCheckBox
from PyQt5.QtCore import Qt

import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt


class ui_plotter(QMainWindow):
	def __init__(self,data,spots,bg=None,parent=None,):
		super(QMainWindow,self).__init__(parent)
		self.gui = parent.gui
		self.ui = plotter(data,spots,bg,self)
		self.setCentralWidget(self.ui)
		self.show()

	def closeEvent(self,event):
		self.parent().activateWindow()
		self.parent().raise_()
		self.parent().setFocus()

class plotter(QWidget):
	def __init__(self,data,spots,bg=None,parent=None):
		super(QWidget,self).__init__(parent=parent)

		self.gui = parent.gui
		layout = QVBoxLayout()

		self.d = data
		self.spots = spots
		self.b = bg

		order = self.cross_corr_order()
		self.d = self.d[order]
		self.spots = self.spots[:,order]
		if not self.b is None:
			self.b = self.b[order]

		self.index = 0

		self.f,self.a = plt.subplots(2,sharex=True)
		self.canvas = FigureCanvas(self.f)
		self.toolbar = NavigationToolbar(self.canvas,None)

		self.slider_select = QSlider(Qt.Horizontal)
		self.slider_select.setMinimum(0)
		self.slider_select.setMaximum(self.d.shape[0]-1)
		self.slider_select.setValue(self.index)

		sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
		self.slider_select.setSizePolicy(sizePolicy)

		self.button_export = QPushButton('Export')
		self.checkbox_spot = QCheckBox('Plot Spot')

		sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
		self.canvas.setSizePolicy(sizePolicy)

		self.a[0].plot(np.random.rand(self.d.shape[0]),color='g',alpha=.8,lw=1)
		self.a[0].plot(np.random.rand(self.d.shape[0]),color='r',alpha=.8,lw=1)
		# if not self.b is None:
			# self.a[0].plot(np.random.rand(self.d.shape[0]),color='g',alpha=.5,lw=1,ls='--')
			# self.a[0].plot(np.random.rand(self.d.shape[0]),color='r',alpha=.5,lw=1,ls='--')
		self.a[1].plot(np.random.rand(self.d.shape[0]),color='b',alpha=.8,lw=1)


		self.label_axes()

		self.canvas.draw()
		# plt.close(self.f)

		twidg = QWidget()
		hbox = QHBoxLayout()
		hbox.addWidget(self.slider_select)
		hbox.addWidget(self.checkbox_spot)
		hbox.addWidget(self.button_export)
		twidg.setLayout(hbox)


		layout.addWidget(self.canvas)
		layout.addWidget(self.toolbar)
		layout.addWidget(twidg)
		self.setLayout(layout)

		self.update()
		self.f.tight_layout()
		self.f.canvas.draw()

		self.init_shortcuts()
		self.slider_select.valueChanged.connect(self.slide_switch)
		self.button_export.clicked.connect(self.export)

	def export(self):
		n = self.gui.data.ncolors

		dd = self.d.copy()
		if n == 2:
			dd[:,1] -= self.gui.prefs['bleedthrough']*dd[:,0]

		q = np.zeros((dd.shape[0]*n,dd.shape[2]))
		for i in range(n):
			q[i::n] = dd[:,i]

		oname = QFileDialog.getSaveFileName(self, 'Export Traces', self.gui.data.filename[:-4]+'_traces.dat','*.dat')
		if oname[0] != "":
			try:
				np.savetxt(oname[0],q.T,delimiter=',')
			except:
				QMessageBox.critical(self,'Export Traces','There was a problem trying to export the traces')

	def label_axes(self):
		self.a[0].set_ylabel(r'$Intensity (a.u.)$',fontsize=14)
		self.a[1].set_ylabel(r'$E_{\rm{FRET}}$',fontsize=14)
		self.a[1].set_xlabel(r'$Time (s)$',fontsize=14)

	def slide_switch(self,v):
		self.index = v
		self.update()

	def plot_spot(self):
		if self.checkbox_spot.checkState():
			self.gui.plot.clear_collections()
			self.ts = self.gui.docks['transform'][1].transforms
			regions,shifts = self.gui.data.regions_shifts()

			# xys = [self.spots[:,np.nonzero(np.arange(self.spots.shape[1]) != self.index)[0]] for _ in range(self.gui.data.ncolors)]
			# for j in range(1,self.gui.data.ncolors):
			# 	xys[j] = self.ts[j][0](xys[j].T).T + shifts[j][:,None]
			#
			# self.gui.plot.scatter(xys[0][0],xys[0][1],radius=.66,color='red')
			# self.gui.plot.scatter(xys[1][0],xys[1][1],radius=.66,color='red')

			xys = [self.spots[:,self.index] for _ in range(self.gui.data.ncolors)]
			for j in range(1,self.gui.data.ncolors):
				xys[j] = self.ts[j][0](xys[j].reshape((1,2))).T[:,0] + shifts[j]
			xys = np.array(xys)
			self.gui.plot.scatter(xys[:,0],xys[:,1],radius=1.25,color='yellow')
			self.gui.plot.canvas.draw()

	def init_shortcuts(self):
		self.make_shortcut(Qt.Key_Left,lambda : self.key('left'))
		self.make_shortcut(Qt.Key_Right,lambda : self.key('right'))

	def make_shortcut(self,key,fxn):
		qs = QShortcut(self)
		qs.setKey(key)
		qs.activated.connect(fxn)

	def key(self,kk):
		if kk == 'right':
			self.index += 1
		elif kk == 'left':
			self.index -= 1
		if self.index < 0:
			self.index = 0
		elif self.index >= self.d.shape[0]:
			self.index = self.d.shape[0]-1
		self.slider_select.setValue(self.index)
		# self.update()

	def update(self):
		pbp = self.gui.prefs['bleedthrough']
		donor = self.d[self.index,0]
		acceptor = self.d[self.index,1] - pbp*self.d[self.index,0]
		t = np.arange(donor.size)*self.gui.prefs['tau']

		self.a[0].lines[0].set_data(t,donor)
		self.a[0].lines[1].set_data(t,acceptor)
		# if not self.b is None:
		# 	bdonor = self.b[self.index,0]
		# 	bacceptor = self.b[self.index,1]
		# 	self.a[0].lines[0].set_data(np.arange(donor.size),bdonor)
		# 	self.a[0].lines[1].set_data(np.arange(donor.size),bacceptor)

		self.a[1].lines[0].set_data(t,acceptor/(donor+acceptor+1e-300))

		self.a[0].set_xlim(0,t[-1])
		self.a[1].set_ylim(-.5,1.5)
		mmin = np.min((donor.min,acceptor.min()))
		mmax = np.max((donor.max(),acceptor.max()))
		delta = mmax-mmin
		self.a[0].set_ylim( mmin - delta*.25, mmax + delta*.25)

		self.a[0].set_title(str(self.index)+' / '+str(self.d.shape[0] - 1))
		self.plot_spot()
		self.canvas.draw()


	def cross_corr_order(self):
		x = self.d[:,0] - self.d[:,0].mean(1)[:,None]
		y = self.d[:,1] - self.d[:,1].mean(1)[:,None]
		a = np.fft.fft(x,axis=1)
		b = np.conjugate(np.fft.fft(y,axis=1))
		b = np.fft.fft(y,axis=1)
		order = np.fft.ifft((a*b),axis=1)
		order = order[:,0].real.argsort()
		return order
