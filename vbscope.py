from PyQt5.QtWidgets import QApplication,QDesktopWidget,QDockWidget, QAction
from PyQt5.QtCore import Qt

import sys
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')

from src.ui import movie_viewer
from src import docks
from src.containers.data import data_container

default_prefs = {
	'render_title':'vbscope',
	'render_artist':'vbscope',
	'plot_fontsize':12
}

class vbscope_gui(movie_viewer):
	def __init__(self,app=None):
		self.app = app

		super(vbscope_gui,self).__init__(self.app)

		self._prefs.combine_prefs(default_prefs)

		self.setup_vbscope_docks()
		self.setup_vbscope_menus()

		self.app_name = 'vbscope'
		self.about_text = "From the Gonzalez Lab (Columbia University).\n\nPrinciple authors: JH,CKT,RLG.\nMany thanks to the entire lab for their input."

		self._prefs.add_pref('ui_width',(QDesktopWidget().availableGeometry(self).size() * 0.7).width())
		self._prefs.add_pref('ui_height',(QDesktopWidget().availableGeometry(self).size() * 0.7).height())

		self.ui_update()
		self.show()


	def setup_vbscope_docks(self):
		## Add Docks
		self.add_dock('tag_viewer', 'Tag Viewer', docks.tag_viewer.dock_tagviewer(self), 'tb', 't')
		self.add_dock('spotfind', 'Spot Find', docks.spotfind.dock_spotfind(self), 'lr', 'r')
		self.add_dock('background', 'Background', docks.background.dock_background(self), 'lr', 'r')
		self.add_dock('transform', 'Transform', docks.transform.dock_transform(self), 'lr', 'r')
		self.add_dock('extract', 'Extract', docks.extract.dock_extract(self), 'lr', 'r')
		self.add_dock('mesoscopic', 'Mesoscopic', docks.mesoscopic.dock_mesoscopic(self), 'lr', 'r')

		## Display docks in tabs
		self.tabifyDockWidget(self.docks['rotate'][0],self.docks['render'][0])
		self.tabifyDockWidget(self.docks['spotfind'][0],self.docks['background'][0])
		self.tabifyDockWidget(self.docks['background'][0],self.docks['transform'][0])
		self.tabifyDockWidget(self.docks['transform'][0],self.docks['extract'][0])
		self.tabifyDockWidget(self.docks['extract'][0],self.docks['mesoscopic'][0])

		## Hide/show certain docks
		self.docks['tag_viewer'][0].hide()
		self.docks['mesoscopic'][0].hide()
		self.docks['tools'][0].raise_()
		self.docks['spotfind'][0].raise_()


	def setup_vbscope_menus(self):
		## Add menu items
		menu_analysis = self.menubar.addMenu('Analysis')
		m = ['spotfind', 'background', 'transform', 'extract', 'mesoscopic']
		for mm in m:
			menu_analysis.addAction(self.docks[mm][0].toggleViewAction())
		self.menu_movie.addAction(self.docks['tag_viewer'][0].toggleViewAction())

	def load_tif(self):
		self.docks['play'][1].stop_playing()

		## Load File
		fname = QFileDialog.getOpenFileName(self,'Choose Movie to load','./')
		if fname[0] != "":
			## Try loading data
			d = data_container(self)
			success = d.load(fname[0])

			if success:
				## Get data
				self.data = d
				self.plot.background = np.zeros_like(self.data.movie[0])

				## Setup plot
				x,y = self.data.movie.shape[1:]
				self.plot.image.set_extent([0,x,0,y])
				self.plot.ax.set_xlim(0,x)
				self.plot.ax.set_ylim(0,y)
				self.plot.ax.set_visible(True) # Turn on the plot -- first initialization
				self.plot.canvas.draw() # Need to initialize on first showing -- for fast plotting

				## Setup play docks
				self.docks['play'][1].slider_frame.setMaximum(self.data.total_frames)
				self.docks['play'][1].slider_frame.setValue(self.data.current_frame+1)
				self.docks['play'][1].update_frame_slider()
				self.docks['play'][1].update_label()

				## Show Image
				self.plot.image.set_data(self.data.movie[self.data.current_frame])
				self.docks['contrast'][1].slider_ceiling.setValue(10000.)
				self.docks['contrast'][1].slider_floor.setValue(0.)
				self.docks['contrast'][1].update_image_contrast()
				self.docks['spotfind'][1].setup_sliders()
				self.plot.image.set_cmap(self.prefs['color map'])
				self.plot.draw()

				## Update tables
				self._prefs.update_table()
				self.docks['tag_viewer'][1].init_model()

			else:
				QMessageBox.critical(None,'Could Not Load File','Could not load file: %s.\nMake sure to use a .TIF format file'%(fname[0]))


def launch():
	app = QApplication([])
	app.setStyle('fusion')
	g = vbscope_gui(app)

	if __name__ == '__main__':
		sys.exit(app.exec_())
	else:
		return g

if __name__ == '__main__':
	launch()
