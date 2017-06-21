from PyQt5.QtWidgets import QMainWindow, QWidget, QMessageBox, QFileDialog, QAction, QApplication, QVBoxLayout, QHBoxLayout, QSizePolicy,  QDockWidget, QDesktopWidget
from PyQt5.QtCore import Qt

import numpy as np

from container_data import data_container
from container_plot import plot_container
import docks

from prefs import default as prefs_default

class gui(QMainWindow):
	def __init__(self,app=None):
		super(QMainWindow,self).__init__()
		self.app = app
		self.initialize()

	def initialize(self):
		self.prefs = prefs_default
		self.data  = data_container(self)
		self.plot  = plot_container()

		self.setCentralWidget(self.plot.canvas)

		self.setup_docks()
		self.setup_menus()
		self.setup_statusbar()

		self.resize(QDesktopWidget().availableGeometry(self).size() * 0.7)
		self.setStyleSheet('background-color: white');
		self.setWindowTitle('vbscope')

		self.show()

	def setup_statusbar(self):
		self.statusbar = self.statusBar()
		self.statusbar.showMessage('Ready')

	def add_dock(self,name,title,widget,areas,loc):
		self.docks[name] = [QDockWidget(title, self), widget]
		if areas == 'lr':
			ar = Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
		elif areas == 'tb':
			ar = Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea
		self.docks[name][0].setAllowedAreas(ar)
		self.docks[name][0].setWidget(self.docks[name][1])
		if loc == 't':
			l = Qt.TopDockWidgetArea
		elif loc == 'b':
			l = Qt.BottomDockWidgetArea
		elif loc == 'l':
			l = Qt.LeftDockWidgetArea
		elif loc == 'r':
			l = Qt.RightDockWidgetArea
		self.addDockWidget(l, self.docks[name][0])

	def setup_docks(self):
		# self.docks is a dictionary with [QDockWidget,Widget] for the docks
		self.docks = {}

		self.add_dock('tools', 'Plot Toolbar', self.plot.toolbar, 'tb', 't')
		self.add_dock('contrast', 'Contrast', docks.contrast.dock_contrast(self), 'tb', 't')
		self.add_dock('play', 'Play', docks.play.dock_play(self), 'tb', 't')
		self.add_dock('rotate', 'Rotate', docks.rotate.dock_rotate(self), 'tb', 't')

		self.add_dock('spotfind', 'Spot Find', docks.spotfind.dock_spotfind(self), 'lr', 'r')
		self.add_dock('background', 'Background', docks.background.dock_background(self), 'lr', 'r')
		self.add_dock('transform', 'Transform', docks.transform.dock_transform(self), 'lr', 'r')
		self.add_dock('extract', 'Extract', docks.extract.dock_extract(self), 'lr', 'r')
		self.add_dock('prefs', 'Prefs', docks.prefs.dock_prefs(self), 'lr', 'r')

		# Display docks in tabs
		self.tabifyDockWidget(self.docks['tools'][0],self.docks['contrast'][0])
		self.tabifyDockWidget(self.docks['contrast'][0],self.docks['play'][0])
		self.tabifyDockWidget(self.docks['play'][0],self.docks['rotate'][0])
		self.docks['tools'][0].raise_()

		self.tabifyDockWidget(self.docks['spotfind'][0],self.docks['background'][0])
		self.tabifyDockWidget(self.docks['background'][0],self.docks['transform'][0])
		self.tabifyDockWidget(self.docks['transform'][0],self.docks['extract'][0])
		self.tabifyDockWidget(self.docks['extract'][0],self.docks['prefs'][0])

		self.docks['background'][0].close()
		self.docks['prefs'][0].close()
		self.docks['spotfind'][0].raise_()


	def setup_menus(self):
		self.menubar = self.menuBar()
		self.menubar.setNativeMenuBar(False)

		### File
		menu_file = self.menubar.addMenu('File')

		file_load = QAction('Load', self, shortcut='Ctrl+O')
		file_load.triggered.connect(self.load_tif)

		file_exit = QAction('Exit', self, shortcut='Ctrl+Q')
		file_exit.triggered.connect(self.app.quit)

		for f in [file_load,file_exit]:
			menu_file.addAction(f)

		### Movie
		menu_movie = self.menubar.addMenu('Movie')
		m = ['tools','contrast','play','rotate']
		for mm in m:
			menu_movie.addAction(self.docks[mm][0].toggleViewAction())

		menu_analysis = self.menubar.addMenu('Analysis')
		m = ['spotfind','background','transform','extract','prefs']
		for mm in m:
			menu_analysis.addAction(self.docks[mm][0].toggleViewAction())

		### Other
		menu_other = self.menubar.addMenu('Other')


	def load_tif(self):
		self.docks['play'][1].stop_playing()
		fname = QFileDialog.getOpenFileName(self,'Choose Movie to load','./')#,filter='TIF File (*.tif *.TIF)')
		if fname[0] != "":
			d = data_container(self)
			success = d.load(fname[0])

			if success:
				self.data = d

				self.plot.background = np.zeros_like(self.data.movie[0])

				x,y = self.data.movie.shape[1:]
				self.plot.image.set_extent([0,x,0,y])
				self.plot.ax.set_xlim(0,x)
				self.plot.ax.set_ylim(0,y)

				self.plot.ax.set_visible(True) # Turn on the plot -- first initialization
				self.plot.canvas.draw() # Need to initialize on first showing -- for fast plotting

				self.docks['play'][1].slider_frame.setMaximum(self.data.total_frames)
				self.docks['play'][1].slider_frame.setValue(self.data.current_frame+1)
				self.docks['play'][1].update_frame_slider()
				self.docks['play'][1].update_label()

				self.plot.image.set_data(self.data.movie[self.data.current_frame])

				self.docks['contrast'][1].slider_ceiling.setValue(10000.)
				self.docks['contrast'][1].slider_floor.setValue(0.)
				self.docks['contrast'][1].update_image_contrast()

				# self.docks['background'][1].combo_method.setCurrentIndex(1)

				self.docks['spotfind'][1].setup_sliders()

				self.plot.image.set_cmap(self.prefs['color map'])
				self.plot.draw()

				self.statusbar.showMessage('Loaded %s'%(self.data.dispname))
				self.setWindowTitle('vbscope - %s'%(self.data.dispname))
				self.prefs['filename'] = self.data.filename
				self.docks['prefs'][1].update_table()
			else:
				QMessageBox.critical(None,'Could Not Load File','Could not load file: %s.\nMake sure to use a .TIF format file'%(fname[0]))

	def closeEvent(self,event):
		self.docks['play'][1].timer_playing.stop()
		# self.main.movie_player.timer_playing.stop()
		pass

def launch():
	import sys
	app = QApplication([])
	app.setStyle('fusion')
	g = gui(app)
	app.setWindowIcon(g.windowIcon())
	sys.exit(app.exec_())

if __name__ == '__main__':
	launch()
