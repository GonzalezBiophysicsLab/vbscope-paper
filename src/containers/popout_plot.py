from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from matplotlib.widgets import  RectangleSelector

from PyQt5.QtWidgets import QSizePolicy,QVBoxLayout,QWidget,QToolBar,QAction,QHBoxLayout,QPushButton,QMainWindow,QDockWidget
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap
import numpy as np
# from ..ui import ui_prefs
from ..ui.ui_prefs import preferences


class popout_plot_container(QMainWindow):
	def __init__(self,nplots_x=1, nplots_y=1, parent=None):
		super(QMainWindow,self).__init__(parent)
		self.ui = popout_plot_container_widget(nplots_x, nplots_y, self)
		self.setCentralWidget(self.ui)
		# self.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
		self.show()

	def closeEvent(self,event):
		self.parent().activateWindow()
		self.parent().raise_()
		self.parent().setFocus()

	def resizeEvent(self,event):
		pass


class popout_plot_container_widget(QWidget):
	def __init__(self,nplots_x=1, nplots_y=1, parent=None):
		super(QWidget,self).__init__()

		self._prefs = preferences(self)
		self.prefs = {
			'fig_width':4.0,
			'fig_height':3.0,
			'label_fontsize':14,
			'label_ticksize':12,
			'label_padding':.1
		}
		self._prefs.update_table()

		self._prefs.edit_callback = self.replot

		self.qd_prefs = QDockWidget("Preferences",self)
		self.qd_prefs.setWidget(self._prefs)
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
		self.canvas.setSizePolicy(sp_fixed)
		self.setSizePolicy(sp_exp)

		self.timer = None

		self.f.set_dpi(self.f.get_dpi()/self.canvas.devicePixelRatio())
		# self.resize_fig()

		self.canvas.draw()
		plt.close(self.f)

		qw = QWidget()
		hbox = QHBoxLayout()
		self.button_refresh = QPushButton("Refresh")
		button_prefs = QPushButton("Preferences")
		self.button_refresh.clicked.connect(self.replot)
		button_prefs.clicked.connect(self.open_preferences)
		hbox.addWidget(self.button_refresh)
		hbox.addWidget(button_prefs)

		hbox.addStretch(1)
		qw.setLayout(hbox)

		self.vbox = QVBoxLayout()
		self.vbox.addWidget(self.canvas)
		self.vbox.addStretch(1)
		self.vbox.addWidget(self.toolbar)
		self.vbox.addWidget(qw)
		# layout.addStretch(1)
		self.setLayout(self.vbox)
		self.f.tight_layout()


	def open_preferences(self):
		try:
			if not self.qd_prefs.isVisible():
				self.qd_prefs.setVisible(True)
			self.qd_prefs.raise_()
		except:
			self.qd_prefs.show()


	def fix_ax(self):
		offset = .08
		offset2 = 0.14
		self.f.subplots_adjust(left=offset2,right=1.-offset,top=1.-offset,bottom=offset2)
		for aa in self.ax:
			aa.tick_params(labelsize=self.prefs['label_ticksize']/self.canvas.devicePixelRatio(),axis='both',direction='in',width=1.0/self.canvas.devicePixelRatio(),length=4./self.canvas.devicePixelRatio())

			aa.tick_params(axis='both', which='major', labelsize=self.prefs['label_ticksize']/self.canvas.devicePixelRatio())
			aa.format_coord = lambda x, y: ''


	def clf(self):
		self.f.clf()
		# self.ax = np.array([self.f.add_subplot(self.nplots,1,i+1) for i in range(self.nplots)])
		self.ax = np.array([[self.f.add_subplot(self.nplots_x,self.nplots_y,j*self.nplots_x + i + 1) for i in range(self.nplots_x)] for j in range(self.nplots_y)])
		if np.any(np.array(self.ax.shape) == 1):
			self.ax = self.ax.reshape(self.ax.size)
		self.fix_ax()

	def resizeEvent(self,event):
		if self.timer is None:
			self.timer = QTimer()
			self.timer.timeout.connect(self._delayreplot)
			self.timer.start(1000)

	def _delayreplot(self):
		self.timer.stop()
		self.timer = None
		self.replot()


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




	def replot(self):
		''' overload me '''
		# print "overload me"

	def setcallback(self,fxn):
		self.replot = fxn
		self._prefs.edit_callback = self.replot
		self.button_refresh.clicked.connect(self.replot)
#
