from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from matplotlib.widgets import  RectangleSelector

from PyQt5.QtWidgets import QSizePolicy,QVBoxLayout,QWidget,QToolBar,QAction,QHBoxLayout,QPushButton,QMainWindow,QDockWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import numpy as np
from ..ui.ui_prefs import preferences


class popout_plot_container(QMainWindow):
	def __init__(self,nplots_x=1, nplots_y=1, parent=None):
		super(QMainWindow,self).__init__(parent)
		self.ui = popout_plot_container_widget(nplots_x, nplots_y, self)
		self.setCentralWidget(self.ui)
		# self.resizeDocks([self.ui.qd_prefs],[200],Qt.Horizontal)
		self.show()

	def closeEvent(self,event):
		self.parent().activateWindow()
		self.parent().raise_()
		self.parent().setFocus()

	# def resizeEvent(self,event):
		# pass


class popout_plot_container_widget(QWidget):
	def __init__(self,nplots_x=1, nplots_y=1, parent=None):
		super(QWidget,self).__init__()

		self.prefs = preferences(self)
		self.prefs.add_dictionary({
			'fig_width':4.0,
			'fig_height':3.0,

			'label_fontsize':8.0,
			'ylabel_offset':-0.165,
			'xlabel_offset':-0.25,
			'font':'Arial',
			'axes_linewidth':1.0,
			'axes_topright':False,
			'tick_fontsize':8.0,
			'tick_length_minor':2.0,
			'tick_length_major':4.0,
			'tick_linewidth':1.0,
			'tick_direction':'out',
			'subplots_left':0.125,
			'subplots_right':0.99,
			'subplots_top':0.99,
			'subplots_bottom':0.155,
			'subplots_hspace':0.04,
			'subplots_wspace':0.03
		})

		self.prefs.edit_callback = self.replot

		self.qd_prefs = QDockWidget("Preferences",self)
		self.qd_prefs.setWidget(self.prefs)
		self.qd_prefs.setAllowedAreas( Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
		parent.addDockWidget(Qt.RightDockWidgetArea, self.qd_prefs)
		# self.qd_prefs.setFloating(True)

		self.nplots_x = nplots_x
		self.nplots_y = nplots_y

		# self.f,self.ax = plt.subplots(nplots_x, nplots_y, sharex=True, figsize=(self.prefs['fig_width']/QPixmap().devicePixelRatio(),self.prefs['fig_height']/QPixmap().devicePixelRatio()))
		self.f,self.ax = plt.subplots(nplots_x, nplots_y, figsize=(self.prefs['fig_width']/QPixmap().devicePixelRatio(),self.prefs['fig_height']/QPixmap().devicePixelRatio()))
		if not type(self.ax) is np.ndarray:
			self.ax = np.array([self.ax])
		self.canvas = FigureCanvas(self.f)
		self.toolbar = NavigationToolbar(self.canvas,None)

		sp_fixed= QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
		sp_exp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
		# self.canvas.setSizePolicy(sp_fixed)
		# self.setSizePolicy(sp_exp)

		self.timer = None

		self.f.set_dpi(self.f.get_dpi()/self.canvas.devicePixelRatio())
		# self.resize_fig()

		self.canvas.draw()
		plt.close(self.f)

		qw = QWidget()
		self.buttonbox = QHBoxLayout()
		self.button_refresh = QPushButton("Refresh")
		button_prefs = QPushButton("Preferences")
		self.button_refresh.clicked.connect(self.replot)
		button_prefs.clicked.connect(self.open_preferences)
		self.buttonbox.addWidget(self.button_refresh)
		self.buttonbox.addWidget(button_prefs)

		self.buttonbox.addStretch(1)
		qw.setLayout(self.buttonbox)

		self.vbox = QVBoxLayout()
		self.vbox.addWidget(self.canvas)
		self.vbox.addStretch(1)
		self.vbox.addWidget(self.toolbar)
		self.vbox.addWidget(qw)
		# layout.addStretch(1)
		self.setLayout(self.vbox)
		self.f.tight_layout()

		self.canvas.resizeEvent = lambda e: e.ignore()
		self.f.resizeEvent = lambda e: e.ignore()
		self.resizeEvent = lambda e: e.ignore()


	def open_preferences(self):
		try:
			if not self.qd_prefs.isVisible():
				self.qd_prefs.setVisible(True)
			self.qd_prefs.raise_()
		except:
			self.qd_prefs.show()


	def fix_ax(self):
		pp = self.prefs
		self.f.subplots_adjust(left=pp['subplots_left'],right=pp['subplots_right'],top=pp['subplots_top'],bottom=pp['subplots_bottom'],hspace=pp['subplots_hspace'],wspace=pp['subplots_wspace'])

		for aa in self.ax.flatten():
			dpr = self.f.canvas.devicePixelRatio()
			for asp in ['top','bottom','left','right']:
				aa.spines[asp].set_linewidth(pp['axes_linewidth']/dpr)
				if asp in ['top','right']:
					aa.spines[asp].set_visible(pp['axes_topright'])

				tickdirection = pp['tick_direction']
				if not tickdirection in ['in','out']: tickdirection = 'in'
				aa.tick_params(labelsize=pp['tick_fontsize']/dpr, axis='both', direction=tickdirection , width=pp['tick_linewidth']/dpr, length=pp['tick_length_minor']/dpr)
				aa.tick_params(axis='both',which='major',length=pp['tick_length_major']/dpr)
				for label in aa.get_xticklabels():
					label.set_family(pp['font'])
				for label in aa.get_yticklabels():
					label.set_family(pp['font'])

	def clf(self):
		self.f.clf()
		# self.ax = np.array([self.f.add_subplot(self.nplots,1,i+1) for i in range(self.nplots)])
		self.ax = np.array([[self.f.add_subplot(self.nplots_x,self.nplots_y,j*self.nplots_x + i + 1) for i in range(self.nplots_x)] for j in range(self.nplots_y)])
		if np.any(np.array(self.ax.shape) == 1):
			self.ax = self.ax.reshape(self.ax.size)
		self.fix_ax()

	# def resizeEvent(self,event):
	# 	if self.timer is None:
	# 		self.timer = QTimer()
	# 		self.timer.timeout.connect(self._delayreplot)
	# 		self.timer.start(1000)
	#
	# def _delayreplot(self):
	# 	self.timer.stop()
	# 	self.timer = None
	# 	self.replot()


	def resize_fig(self):

		self.f.clf()
		self.canvas.resize(int(self.prefs['fig_width']*self.f.get_dpi()/self.canvas.devicePixelRatio()),int(self.prefs['fig_height']*self.f.get_dpi()/self.canvas.devicePixelRatio()))
		self.f.set_figheight(self.prefs['fig_height']/self.canvas.devicePixelRatio())
		self.f.set_figwidth(self.prefs['fig_width']/self.canvas.devicePixelRatio())
		self.f.set_size_inches(self.prefs['fig_width']/self.canvas.devicePixelRatio(),self.prefs['fig_height']/self.canvas.devicePixelRatio())

		# self.resize(int(self.prefs['fig_width']*self.f.get_dpi()/self.canvas.devicePixelRatio()),int(self.prefs['fig_height']*self.f.get_dpi()/self.canvas.devicePixelRatio()))

		# self.ax = np.array([self.f.add_subplot(self.nplots,1,i+1) for i in range(self.nplots)])
		self.ax = np.array([[self.f.add_subplot(self.nplots_x,self.nplots_y,j*self.nplots_x + i + 1) for i in range(self.nplots_x)] for j in range(self.nplots_y)])
		if np.any(np.array(self.ax.shape) == 1):
			self.ax = self.ax.reshape(self.ax.size)

		self.fix_ax()
		self.canvas.update()
		self.canvas.flush_events()
		self.canvas.draw()

	def figure_out_ticks(self,ymin,ymax,nticks):
		m = nticks
		if m <= 0: return ()
		if ymax <= ymin: return ()

		delta = np.abs(ymax-ymin)
		o = np.floor(np.log10(delta))
		d = 10.**o
		y0 = np.ceil(ymin/d)*d
		delta = np.abs(ymax-y0)

		if delta <= d:
			d /= 10.
			y0 = np.ceil(ymin/d)*d
			delta = np.abs(ymax-y0)
		n = np.floor(delta/d)
		f = 2.**(np.floor(np.log2(n/m))+1)
		d*=f

		if d<=0: return ()

		y0 = np.ceil(ymin/d)*d
		delta = np.abs(ymax-y0)
		n = np.floor(delta/d)

		ticks = np.linspace(y0,y0+n*d,n+1)
		return ticks



	def replot(self):
		''' overload me '''
		# print "overload me"

	def setcallback(self,fxn):
		self.replot = fxn
		self.prefs.edit_callback = self.replot
		self.button_refresh.clicked.connect(self.replot)
#
