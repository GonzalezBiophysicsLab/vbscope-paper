from PyQt5.QtWidgets import QMessageBox,QFileDialog,QAction,QShortcut,QSizePolicy
#
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence

from ui_general import gui

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')

from ..containers import data_container,plot_container
from .. import docks

default_prefs = {
	'movie_playback_fps':100,
	'movie_tau':1.,

	'plot_colormap':'Greys_r',
	'plot_contrast_scale':20.,

	'render_renderer':'ffmpeg',
	'render_title':'Movie Render',
	'render_artist':'Movie Viewer',
	'render_fps':100,
	'render_codec':'h264'
}

class movie_viewer(gui):
	def __init__(self,app=None,flag_min_docks=False):
		self.app = app
		self.data  = data_container(self)
		self.plot  = plot_container()

		super(movie_viewer,self).__init__(self.app,self.plot.canvas)

		self.app_name = 'Movie Viewer'
		self.setWindowTitle(self.app_name)

		self.flag_min_docks = flag_min_docks
		self.initialize()

	def initialize(self):

		self._prefs.combine_prefs(default_prefs)

		self.setup_docks()
		self.setup_menus()
		self.setup_shortcuts()

		self.show()

	def setup_shortcuts(self):
		self.shortcut_esc = QShortcut(QKeySequence(Qt.Key_Escape),self,self.plot.remove_rectangle)

	def setup_docks(self):
		self.add_dock('tools', 'Plot Toolbar', self.plot.toolbar, 'tb', 't')
		self.add_dock('contrast', 'Contrast', docks.contrast.dock_contrast(self), 'tb', 't')
		self.add_dock('play', 'Play', docks.play.dock_play(self), 'tb', 't')
		self.add_dock('rotate', 'Rotate', docks.rotate.dock_rotate(self), 'tb', 't')
		if not self.flag_min_docks:
			self.add_dock('render', 'Render Movie', docks.render.dock_render(self), 'tb', 't')

		# Display docks in tabs
		self.tabifyDockWidget(self.docks['tools'][0],self.docks['contrast'][0])
		self.tabifyDockWidget(self.docks['contrast'][0],self.docks['play'][0])
		self.tabifyDockWidget(self.docks['play'][0],self.docks['rotate'][0])
		if not self.flag_min_docks:

			self.tabifyDockWidget(self.docks['rotate'][0],self.docks['render'][0])
		self.docks['tools'][0].raise_()

	def setup_menus(self):
		self.menu_movie = self.menubar.addMenu('Movie')
		m = ['tools','contrast','play','rotate']
		if not self.flag_min_docks:
			[m.append(mm) for mm in ['render']]
		for mm in m:
			self.menu_movie.addAction(self.docks[mm][0].toggleViewAction())


	def load(self,event=None,fname=None):
		self.docks['play'][1].stop_playing()
		if fname is None:
			fname = QFileDialog.getOpenFileName(self,'Choose Movie to load','./')
		else:
			fname = [fname]
		if fname[0] != "":
			d = data_container(self)
			success = d.load(fname[0])

			if success:
				self.data = d
				# print self.data.total_frames

				y,x = self.data.movie.shape[1:]
				self.plot.image.set_extent([-.5,x-.5,-.5,y-.5])
				self.plot.ax.set_xlim(-.5,x-.5)
				self.plot.ax.set_ylim(-.5,y-.5)

				self.plot.ax.set_visible(True) # Turn on the plot -- first initialization
				self.plot.canvas.draw() # Need to initialize on first showing -- for fast plotting

				self.docks['play'][1].slider_frame.setMaximum(self.data.total_frames)
				self.docks['play'][1].slider_frame.setValue(self.data.current_frame+1)
				self.docks['play'][1].update_frame_slider()
				self.docks['play'][1].update_label()

				if not self.flag_min_docks:

					self.docks['render'][1].spin_start.setMaximum(self.data.total_frames)
					self.docks['render'][1].spin_end.setMaximum(self.data.total_frames)
					self.docks['render'][1].spin_end.setValue(self.data.total_frames)

				self.plot.image.set_data(self.data.movie[self.data.current_frame])

				self.docks['contrast'][1].slider_ceiling.setValue(65535)
				self.docks['contrast'][1].slider_floor.setValue(0)
				self.docks['contrast'][1].update_image_contrast()

				self.plot.image.set_cmap(self.prefs['plot_colormap'])
				self.plot.draw()

				self.log('Loaded %s'%(self.data.filename),True)

				self.setWindowTitle('%s - %s'%(self.app_name,self.data.filename))
				self._prefs.add_pref('movie_filename',self.data.filename)
				return True

			else:
				message = 'Could not load file: %s.'%(fname[0])
				QMessageBox.critical(None,'Could Not Load File',message)
				self.log(message,True)
		return False

	def ui_update(self):
		self.plot.toolbar.setStyleSheet('color:%s;background-color:%s;'%(self.prefs['ui_fontcolor'],self.prefs['ui_bgcolor']))
		self.plot.f.set_facecolor(self.prefs['ui_bgcolor'])
		self.plot.canvas.draw()
		super(movie_viewer,self).ui_update()
