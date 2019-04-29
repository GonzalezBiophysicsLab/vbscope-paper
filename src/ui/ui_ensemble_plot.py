from PyQt5.QtWidgets import QListWidget,QMainWindow,QWidget,QAbstractItemView,QFileDialog,QPushButton,QVBoxLayout,QHBoxLayout,QLabel,QFormLayout,QLineEdit,QComboBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QPixmap

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import matplotlib.pyplot as plt
import numpy as np

class gui_ensemble_plot(QMainWindow):
	def __init__(self,parent=None):
		super(QMainWindow,self).__init__(parent)
		self.ui = ensemble_plot(self)
		self.setCentralWidget(self.ui)
		self.show()

	def closeEvent(self,event):
		if not self.parent() is None:
			self.parent().raise_()
			self.parent().setFocus()

class ensemble_plot(QWidget):
	def __init__(self,parent=None):
		super(QWidget,self).__init__(parent=parent)
		self.gui= self.parent().parent()
		self.initialize()

	def initialize(self):

		self.combo_plot = QComboBox()
		self.combo_plot.addItems(['N vs time','SBR vs time'])

		self.fig, self.ax = plt.subplots(1, figsize = (4.0/QPixmap().devicePixelRatio(), 2.5/QPixmap().devicePixelRatio()),sharex=True)
		self.canvas = FigureCanvas(self.fig)
		self.toolbar = NavigationToolbar(self.canvas,None)
		self.fig.set_dpi(self.fig.get_dpi()/self.canvas.devicePixelRatio())
		self.canvas.draw()
		plt.close(self.fig)

		self.button_refresh = QPushButton('Refresh')

		#####################################
		#####################################

		wid1 = QWidget()
		hbox = QHBoxLayout()
		hbox.addWidget(QLabel('Plot Type'))
		hbox.addWidget(self.combo_plot)
		hbox.addWidget(self.button_refresh)
		wid1.setLayout(hbox)

		vbox = QVBoxLayout()
		vbox.addWidget(wid1)
		vbox.addStretch(1)
		vbox.addWidget(self.canvas)
		vbox.addWidget(self.toolbar)

		self.setLayout(vbox)

		#####################################
		#####################################

		self.button_refresh.clicked.connect(self.replot)
		self.combo_plot.currentIndexChanged.connect(self.replot)

		self.replot()

	def replot(self):
		self.ax.cla()
		if self.combo_plot.currentIndex() == 0:
			self.plot_nvtime()
		elif self.combo_plot.currentIndex() == 1:
			self.plot_sbrvtime()
		self.canvas.draw()

	def keyPressEvent(self,event):
		if event.key() == Qt.Key_Escape:
			self.gui.open_preferences()
			return
		super(ensemble_plot,self).keyPressEvent(event)

	def plot_nvtime(self):
		sfd = self.gui.docks['spotfind'][1]
		ylim = [0,1]
		xlim = [0,1]
		if not sfd.spotprobs is None:
			if not sfd.spotprobs[0] is None:
				nc = self.gui.data.ncolors
				y = [None,]*nc
				for i in range(nc):
					cc = self.gui.prefs['channels_colors'][i]
					y[i] = (sfd.spotprobs[i] > sfd.pp).sum((1,2))
					self.ax.plot(np.arange(y[i].size)*self.gui.prefs['movie_tau'],y[i],color=cc,lw=1,alpha=.95)
				ylim = [0,np.max([np.max(yy) for yy in y])+10]
				xlim = [0,y[0].size]

		self.ax.set_xlabel('Time (s)')
		self.ax.set_ylabel('Count')
		self.ax.set_xlim(*xlim)
		self.ax.set_ylim(*ylim)
		self.fig.tight_layout()

	def plot_sbrvtime(self):
		sfd = self.gui.docks['spotfind'][1]
		ylim = [0,1]
		xlim = [0,1]
		if not sfd.spotprobs is None:
			if not sfd.spotprobs[0] is None:
				nc = self.gui.data.ncolors
				y = [None,]*nc
				for i in range(nc):
					cc = self.gui.prefs['channels_colors'][i]
					# y[i] = (sfd.spotprobs[i] > sfd.pp).sum((1,2))
					# self.ax.plot(np.arange(y[i].size)*self.gui.prefs['movie_tau'],y[i],color=cc,lw=1,alpha=.95)
				# ylim = [0,np.max([np.max(yy) for yy in y])+5]
				# xlim = [0,y[0].size]

		self.ax.set_xlabel('Time (s)')
		self.ax.set_ylabel('SBR')
		self.ax.set_title('Not implemented yet')
		self.ax.set_xlim(*xlim)
		self.ax.set_ylim(*ylim)
		self.fig.tight_layout()
