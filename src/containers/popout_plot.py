from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from matplotlib.widgets import  RectangleSelector

from PyQt5.QtWidgets import QSizePolicy,QVBoxLayout,QWidget,QToolBar,QAction,QHBoxLayout,QPushButton

import numpy as np
# from ..ui import ui_prefs
from ..ui.ui_prefs import preferences


class popout_plot_container(QWidget):
	def __init__(self,nplots=1):
		super(QWidget,self).__init__()

		self._prefs = preferences(self)
		self.prefs = {}
		self._prefs.update_table()

		self._prefs.edit_callback = self.replot

		self.nplots = nplots

		self.f,self.ax = plt.subplots(nplots,sharex=True)
		if not type(self.ax) is np.ndarray:
			self.ax = np.array([self.ax])
		self.canvas = FigureCanvas(self.f)
		self.toolbar = NavigationToolbar(self.canvas,None)

		# sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
		sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

		self.canvas.setSizePolicy(sizePolicy)
		self.f.set_dpi(self.f.get_dpi()/self.canvas.devicePixelRatio())
		self.fix_ax()

		self.canvas.draw()
		plt.close(self.f)


		qw = QWidget()
		hbox = QHBoxLayout()
		hbox.addWidget(self.toolbar)
		button_refresh = QPushButton("Refresh")
		button_prefs = QPushButton("Preferences")
		button_refresh.clicked.connect(self.replot)
		button_prefs.clicked.connect(self.open_preferences)
		hbox.addWidget(button_refresh)
		hbox.addWidget(button_prefs)

		hbox.addStretch(1)
		qw.setLayout(hbox)

		layout = QVBoxLayout()
		layout.addWidget(self.canvas)
		# layout.addWidget(self.toolbar)
		layout.addWidget(qw)

		# layout.addStretch(1)
		self.setLayout(layout)
		self.f.tight_layout()


	def open_preferences(self):
		self._open_ui(self._prefs)

	def _open_ui(self,ui):
		try:
			if not ui.isVisible():
				ui.setVisible(True)
			ui.raise_()
		except:
			ui.show()
		ui.showNormal()
		ui.activateWindow()

	def fix_ax(self):
		offset = .08
		offset2 = 0.14
		self.f.subplots_adjust(left=offset2,right=1.-offset,top=1.-offset,bottom=offset2)
		for aa in self.ax:
			aa.tick_params(labelsize=12./self.canvas.devicePixelRatio(),axis='both',direction='in',width=1.0/self.canvas.devicePixelRatio(),length=4./self.canvas.devicePixelRatio())

			aa.tick_params(axis='both', which='major', labelsize=12./self.canvas.devicePixelRatio())
			aa.format_coord = lambda x, y: ''


	def clf(self):
		self.f.clf()
		self.ax = np.array([self.f.add_subplot(self.nplots,1,i+1) for i in range(self.nplots)])
		self.fix_ax()

	def resizeEvent(self,event):
		self.fix_ax()

	def replot(self):
		''' overload this '''
		print "overload me"
