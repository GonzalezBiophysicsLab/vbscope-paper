from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QLabel, QComboBox, QSlider,  QLineEdit, QMessageBox, QFileDialog, QShortcut, QAction, QApplication, QGridLayout, QVBoxLayout, QHBoxLayout, QSizePolicy, QGroupBox, QSizePolicy
from PyQt5.QtGui import QKeySequence,QIcon
from PyQt5.QtCore import Qt,QTimer

import numpy as np

class Main(QWidget):
	def __init__(self,parent=None):
		super(Main, self).__init__(parent)

		self.app = self.parent().app

		self.initialize()

	def initialize(self):
		self.setup_variables()

		# ## Menubar
		file_load_action = QAction('Load', self)
		file_load_action.setShortcut('Ctrl+O')
		file_load_action.setStatusTip('load movie')
		file_load_action.triggered.connect(self.load_tif)

		file_exit_action = QAction(QIcon('exit.png'), 'Exit', self)
		file_exit_action.setShortcut('Ctrl+Q')
		file_exit_action.setStatusTip('Exit application')
		file_exit_action.triggered.connect(self.parent().app.quit)

		self.menubar = self.parent().menuBar()
		self.menubar.setNativeMenuBar(False)
		fileMenu = self.menubar.addMenu('File')
		fileMenu.addAction(file_load_action)
		fileMenu.addAction(file_exit_action)

		self.statusbar = self.parent().statusBar()
		self.statusbar.showMessage('Ready')

		##
		overall_layout = QHBoxLayout()

		### LHS
		lhs = QWidget()
		vbox = QVBoxLayout()

		#### Top Bar
		topbar = QWidget()
		hbox_top = QHBoxLayout()
		self.label_filename = QLabel(self.pref['filename'])
		self.label_framenumber = QLabel('%d / %d || '%(self.current_frame+1,self.total_frames))
		self.label_framenumber.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
		self.label_filesize = QLabel('0.0 Mb')
		hbox_top.addWidget(self.label_filename)
		hbox_top.addStretch(1)
		hbox_top.addWidget(self.label_framenumber)
		hbox_top.addWidget(self.label_filesize)
		topbar.setLayout(hbox_top)

		#### MPL Canvas
		self.f,self.ax = plt.subplots(1)
		self.canvas = FigureCanvas(self.f)
		sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
  		self.canvas.setSizePolicy(sizePolicy)
		self.toolbar = NavigationToolbar(self.canvas,None)
		self.image = self.ax.imshow(np.zeros((10,10)),cmap='viridis',interpolation='nearest',origin='lower')
		self.ax.set_visible(False)
		self.f.subplots_adjust(left=.05,right=.95,top=.95,bottom=.05)
		self.canvas.draw()
		plt.close(self.f)


		#### Playbar
		playbar = QWidget()
		hbox_play = QHBoxLayout()
		self.button_play = QPushButton(u"\u25B6")
		self.slider_frame = QSlider(Qt.Horizontal)
		sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
  		self.slider_frame.setSizePolicy(sizePolicy)
		self.slider_frame.setMinimum(1)
		self.slider_frame.setMaximum(1)
		self.slider_frame.setValue(self.current_frame)

		hbox_play.addWidget(self.button_play)
		hbox_play.addWidget(self.slider_frame)
		playbar.setLayout(hbox_play)

		#### Navigation bar
		nav_bar = QWidget()
		hbox_nav = QHBoxLayout()
		box_slide = QGroupBox("Contrast")
		grid = QGridLayout()
		self.slider_ceiling = QSlider(Qt.Horizontal)
		self.slider_floor = QSlider(Qt.Horizontal)
		sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
  		self.slider_ceiling.setSizePolicy(sizePolicy)
		self.slider_floor.setSizePolicy(sizePolicy)
		[ss.setMinimum(0.) for ss in [self.slider_ceiling,self.slider_floor]]
		[ss.setMaximum(10000.) for ss in [self.slider_ceiling,self.slider_floor]]
		self.slider_ceiling.setValue(10000.)
		self.slider_floor.setValue(0.)
		grid.addWidget(QLabel('Floor'),0,0)
		grid.addWidget(QLabel('Ceiling'),1,0)
		grid.addWidget(self.slider_floor,0,1)
		grid.addWidget(self.slider_ceiling,1,1)
		box_slide.setLayout(grid)
		sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
  		box_slide.setSizePolicy(sizePolicy)

		hbox_nav.addWidget(box_slide)
		hbox_nav.addWidget(self.toolbar)
		nav_bar.setLayout(hbox_nav)


		### Compile LHS
		vbox.addWidget(topbar)
		vbox.addWidget(self.canvas)
		vbox.addWidget(nav_bar)
		vbox.addWidget(playbar)

		lhs.setLayout(vbox)
		sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
  		lhs.setSizePolicy(sizePolicy)

		overall_layout.addWidget(lhs)
		self.setLayout(overall_layout)


		self.connect_buttons()

	def setup_variables(self):
		from preferences import default as pref_default
		self.pref = pref_default

		self.movie = None

		self.current_frame = 0
		self.total_frames = 1
		self.flag_playing = False


	def connect_buttons(self):
		self.button_play.clicked.connect(self.play_movie)
		self.timer_playing = QTimer()

		self.slider_frame.valueChanged.connect(self.update_frame_slider)
		self.slider_floor.valueChanged.connect(self.change_floor)
		self.slider_ceiling.valueChanged.connect(self.change_ceiling)


##############################

	def update_image_contrast(self):
		if not self.movie is None:
			xx = self.pref['contrast_scale']*((self.pref['image_contrast'] / 100.) -.5)
			y = 1./(1.+np.exp(-xx))
			self.image.set_clim(np.percentile(self.movie[self.current_frame],y[0]*100.),np.percentile(self.movie[self.current_frame],y[1]*100.))
			self.canvas.restore_region(self.blit_axis)
			self.ax.draw_artist(self.image)
			self.canvas.blit(self.ax.bbox)

	def change_floor(self,v):
		vv = float(v)/100.
		self.pref['image_contrast'][0] = vv
		if vv >= self.pref['image_contrast'][1]:
			self.pref['image_contrast'][0] = self.pref['image_contrast'][1] - 1
			self.slider_floor.setValue(self.pref['image_contrast'][0])
		self.update_image_contrast()

	def change_ceiling(self,v):
		vv = float(v)/100.
		self.pref['image_contrast'][1] = vv
		if vv <= self.pref['image_contrast'][0]:
			self.pref['image_contrast'][1] = self.pref['image_contrast'][0] + 1
			self.slider_ceiling.setValue(self.pref['image_contrast'][1])
		self.update_image_contrast()

	def update_frame_slider(self,value):
		if not self.movie is None:
			self.current_frame = value - 1
			self.update_frame()

	def update_frame(self):
		self.label_framenumber.setText('%d / %d ||'%(self.current_frame+1,self.total_frames))
		self.image.set_data(self.movie[self.current_frame])

		self.canvas.restore_region(self.blit_axis)
		self.ax.draw_artist(self.image)
		self.canvas.blit(self.ax.bbox)
		# self.canvas.draw()
		# self.canvas.flush_events()

	def stop_playing(self):
		self.timer_playing.stop()
		self.flag_playing = False
		self.button_play.setText(u'\u25B6')
		self.timer_playing = QTimer()


	def advance_frame(self):
		if self.current_frame < self.total_frames-1:
			self.current_frame += 1
			self.slider_frame.setValue(self.current_frame)
			self.update_frame()
		else:
			self.stop_playing()

	def play_movie(self):
		if not self.flag_playing and not self.movie is None:

			self.blit_axis = self.canvas.copy_from_bbox(self.ax.bbox)
			self.timer_playing.timeout.connect(self.advance_frame)
			self.flag_playing = True
			self.button_play.setText('||')
			self.timer_playing.start(1./self.pref['playback_fps']*1000)
		else:
			self.stop_playing()

	def setup_movie(self):
		self.ax.set_visible(True)
		self.ax.set_axis_off()
		self.canvas.draw()
		self.blit_axis = self.canvas.copy_from_bbox(self.ax.bbox)
		self.update_image_contrast()
		self.update_frame()
		self.canvas.draw()

	def load_tif(self):
		fname = QFileDialog.getOpenFileName(self,'Choose Movie to load','./')#,filter='TIF File (*.tif *.TIF)')
		data = None
		if fname[0] != "":
			fname = fname[0]
			try:
				from io_movie import io_load_tif
				data = io_load_tif(fname)
			except:
				QMessageBox.critical(None,'Could Not Load File','Could not load file: %s.\nMake sure to use a .TIF file'%fname)

		if not data is None:
			self.pref['filename'] = fname
			if len(self.pref['filename']) > 80:
				dispfname ="....."+self.pref['filename'][-80:]
			else:
				dispfname = self.pref['filename']
			from os.path import getsize
			filesize = int(np.round(getsize(self.pref['filename'])/1e6))
			self.label_filename.setText(dispfname)
			self.label_filesize.setText('%d Mb'%(filesize))
			self.movie = data
			self.total_frames = self.movie.shape[0]
			self.slider_frame.setMaximum(self.total_frames)
			self.setup_movie()
			self.statusbar.showMessage('Loaded %s'%(fname))

class gui(QMainWindow):
	def __init__(self,app=None):
		super(QMainWindow,self).__init__()
		self.app = app
		self.initialize()

	def initialize(self):
		self.main = Main(self)
		self.setCentralWidget(self.main)

		self.setWindowTitle('vbscope')
		self.show()

	def closeEvent(self,event):
		self.main.timer_playing.stop()

def launch():
	import sys
	app = QApplication([])
	app.setStyle('fusion')
	g = gui(app)
	app.setWindowIcon(g.windowIcon())
	sys.exit(app.exec_())

if __name__ == '__main__':
	launch()
