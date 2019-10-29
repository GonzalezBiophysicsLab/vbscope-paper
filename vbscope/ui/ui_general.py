from PyQt5.QtWidgets import QApplication,QMainWindow, QDockWidget, QAction, QMessageBox,QProgressDialog,QMessageBox,QShortcut, QDockWidget, QFileDialog,QDesktopWidget
from PyQt5.QtCore import Qt, qInstallMessageHandler
from PyQt5.QtGui import QKeySequence, QIcon

import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np

from .ui_log import logger
from .ui_prefs import preferences
from .ui_progressbar import progressbar

from ..containers import data_container,plot_container
from .. import docks

# from src.ui import movie_viewer
from .. import docks
from ..containers import data_container
# from src import plots
from .ui_ensemble_plot import gui_ensemble_plot


default_prefs = {
	'movie_playback_fps':100,
	'movie_tau':1.,

	'plot_colormap':'Greys_r',
	'plot_contrast_scale':20.,

	'render_renderer':'ffmpeg',
	'render_title':'Movie Render',
	'render_artist':'Movie Viewer',
	'render_fps':100,
	'render_codec':'h264',
	'render_title':'vbscope',
	'render_artist':'vbscope',
	'plot_fontsize':12,

	'channels_colors':['green','red','blue','purple'],
	'channels_wavelengths':np.array((570.,680.,488.,800.)),
	'calibrate_gain':12.952,
	'calibrate_offset':482.,
}

class vbscope_gui(QMainWindow):
	'''
	UI Objects of Importance:
		* app
		* menubar
		* _log
		* prefs
		* docks

	Functions of Importance:
		* add_dock
		* load_movie
		* set_status
		* log
		* load - overload this
		* prefs.add_dictionary (for a dictionary)
		* prefs['new_pref_name'] = new_pref_value  (for one entry)

	Variables of Importance:
		* app_name
		* about_text
		* prefs
	'''
	## app - the main app
	## menubar - the menubar
	## log - logger
	## prefs - preferences
	## statusbar - status bar

	## add_dock - add a dock

	def __init__(self,app=None):
		super(QMainWindow,self).__init__()
		self.app = app
		self.app_name = "vbscope"
		self.setWindowTitle(self.app_name)

		self.data  = data_container(self)
		self.plot  = plot_container()

		self.setCentralWidget(self.plot.canvas)

		self.closeEvent = self.safe_close

		self._log = logger()

		self.prefs = preferences(self)
		self.prefs.add_dictionary(default_prefs)

		self.qd_prefs = QDockWidget("Preferences",self)
		self.qd_prefs.setWidget(self.prefs)
		self.qd_prefs.setAllowedAreas( Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
		self.addDockWidget(Qt.RightDockWidgetArea, self.qd_prefs)

		# if not flag_floatprefs:
		self.addDockWidget(Qt.LeftDockWidgetArea, self.qd_prefs)
		# else:
		# 	self.qd_prefs.setFloating(True)
		# 	self.qd_prefs.hide()
		# 	self.qd_prefs.topLevelChanged.connect(self.resize_prefs)

		self.init_statusbar()
		self.init_docks()
		self.setup_docks()
		self.init_menus()
		self.setup_vbscope_plots()
		self.setup_vbscope_menus()
		self.setup_shortcuts()

		self.about_text = "From the Gonzalez Lab (Columbia University).\n\nPrinciple authors: JH,CKT,RLG.\nMany thanks to the entire lab for their input."

		self.prefs['ui_width']  = (QDesktopWidget().availableGeometry(self).size() * 0.7).width()
		self.prefs['ui_height'] = (QDesktopWidget().availableGeometry(self).size() * 0.7).height()
		self.move(0,0)

		self.ui_update()
		self.resize_prefs()

	def mousePressEvent(self,event):
		self.setFocus()
		super(QMainWindow,self).mousePressEvent(event)

	def show(self):
		super(vbscope_gui,self).show()
		## Toggle to make prefs show up
		self.resize(self.size().width(),self.size().height()+1)
		self.resize(self.size().width(),self.size().height()-1)

	def setup_vbscope_menus(self):
		self.menu_movie = self.menubar.addMenu('Movie')
		m = ['tools','contrast','play','rotate','render']
		# if not self.flag_min_docks:
			# [m.append(mm) for mm in ['render']]
		for mm in m:
			self.menu_movie.addAction(self.docks[mm][0].toggleViewAction())
		## Add menu items
		menu_analysis = self.menubar.addMenu('Analysis')
		analysis_calibrate = QAction('Convert to electrons',self)
		analysis_calibrate.triggered.connect(self.dark_cal)
		menu_analysis.addAction(analysis_calibrate)
		m = ['spotfind', 'background', 'transform', 'extract']
		for mm in m:
			menu_analysis.addAction(self.docks[mm][0].toggleViewAction())
		# self.menu_movie.addAction(self.docks['tag_viewer'][0].toggleViewAction())

		action_psftool = QAction('PSF Tool',self)
		action_psftool.triggered.connect(self.launch_psftool)
		self.menubar.addAction(action_psftool)

	def setup_vbscope_plots(self):
		# menu_plot = self.menubar.addMenu('Plots')
		# # plt_region = QAction('Region plot',self)
		# # plt_region.triggered.connect(self.plot_region)
		# #
		# # for a in [plt_region]:
		# # 	menu_plot.addAction(a)
		# #
		# # # self.popout_plots = {
		# # # 	'plot_region':None
		# # # }

		# plots_ensemble = QAction('Ensemble Plots',self)
		# plots_ensemble.triggered.connect(self.show_ensemble_plot)
		# menu_plot.addAction(plots_ensemble)
		self.ui_update()

	def show_ensemble_plot(self):
		try:
			if not self.ui_ensemble_plot.isVisible():
				self.ui_ensemble_plot.setVisible(True)
			self.ui_ensemble_plot.raise_()
		except:
			self.ui_ensemble_plot = gui_ensemble_plot(self)
			self.ui_ensemble_plot.setWindowTitle('Ensemble Plots')
			self.ui_ensemble_plot.show()

	def setup_shortcuts(self):
		pass
		# self.shortcut_esc = QShortcut(QKeySequence(Qt.Key_Escape),self,self.plot.remove_rectangle)

	def init_statusbar(self):
		self.statusbar = self.statusBar()
		self.statusbar.showMessage('Initialized')

	def setup_docks(self):
		self.add_dock('tools', 'Plot Toolbar', self.plot.toolbar, 'tb', 't')
		self.add_dock('contrast', 'Contrast', docks.contrast.dock_contrast(self), 'tb', 't')
		self.add_dock('play', 'Play', docks.play.dock_play(self), 'tb', 't')
		self.add_dock('rotate', 'Rotate', docks.rotate.dock_rotate(self), 'tb', 't')
		self.add_dock('render', 'Render Movie', docks.render.dock_render(self), 'tb', 't')

		# Display docks in tabs
		self.tabifyDockWidget(self.docks['tools'][0],self.docks['contrast'][0])
		self.tabifyDockWidget(self.docks['contrast'][0],self.docks['play'][0])
		self.tabifyDockWidget(self.docks['play'][0],self.docks['rotate'][0])
		self.tabifyDockWidget(self.docks['rotate'][0],self.docks['render'][0])
		self.docks['tools'][0].raise_()

		## Add Docks
		self.add_dock('tag_viewer', 'Tag Viewer', docks.tag_viewer.dock_tagviewer(self), 'tb', 't')
		self.add_dock('spotfind', 'Spot Find', docks.spotfind.dock_spotfind(self), 'lr', 'r')
		self.add_dock('background', 'Background', docks.background.dock_background(self), 'lr', 'r')
		self.add_dock('transform', 'Transform', docks.transform.dock_transform(self), 'lr', 'r')
		self.add_dock('extract', 'Extract', docks.extract.dock_extract(self), 'lr', 'r')
		# self.add_dock('mesoscopic', 'Mesoscopic', docks.mesoscopic.dock_mesoscopic(self), 'lr', 'r')

		## Display docks in tabs
		self.tabifyDockWidget(self.docks['rotate'][0],self.docks['render'][0])
		self.tabifyDockWidget(self.docks['spotfind'][0],self.docks['background'][0])
		self.tabifyDockWidget(self.docks['background'][0],self.docks['transform'][0])
		self.tabifyDockWidget(self.docks['transform'][0],self.docks['extract'][0])
		# self.tabifyDockWidget(self.docks['extract'][0],self.docks['mesoscopic'][0])

		## Hide/show certain docks
		self.docks['tag_viewer'][0].hide()
		# self.docks['mesoscopic'][0].hide()
		self.docks['tools'][0].raise_()
		self.docks['spotfind'][0].raise_()

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

		try:
			self.prefs.add_dictionary(widget.default_prefs)

		except:
			pass

	def init_docks(self):
		# self.docks is a dictionary with [QDockWidget,Widget] for the docks
		self.docks = {}

	def init_menus(self):
		self.menubar = self.menuBar()
		self.menubar.setNativeMenuBar(False)

		### File
		self.menu_file = self.menubar.addMenu('File')

		file_load = QAction('Load', self, shortcut='Ctrl+O')
		file_load.triggered.connect(lambda e: self.load())

		file_log = QAction('Log', self,shortcut='F2')
		file_log.setShortcutContext(Qt.ApplicationShortcut)
		file_log.triggered.connect(self.open_log)

		file_prefs = QAction('Preferences', self,shortcut='F3')
		file_prefs.setShortcutContext(Qt.ApplicationShortcut)
		file_prefs.triggered.connect(self.open_preferences)

		file_saveprefs = QAction('Save Preferences',self)
		file_saveprefs.triggered.connect(lambda e: self.prefs.save_preferences())
		file_loadprefs = QAction('Load Preferences',self)
		file_loadprefs.triggered.connect(lambda e: self.prefs.load_preferences())

		self.about_text = ""
		file_about = QAction('About',self)
		file_about.triggered.connect(self.about)

		file_exit = QAction('Exit', self, shortcut='Ctrl+Q')
		# file_exit.triggered.connect(self.app.quit)
		file_exit.triggered.connect(self.close)

		for f in [file_load,file_log,file_prefs,file_loadprefs,file_saveprefs,file_about,file_exit]:
			self.menu_file.addAction(f)

################################################################################

	def load(self,fname=None):
		self.docks['play'][1].stop_playing()
		if fname is None:
			fname = QFileDialog.getOpenFileName(self,'Choose Movie to load','./')
		else:
			fname = [fname]
		if fname[0] != "":
			if fname[0].endswith('.hdf5') or fname[0].endswith('.hdf'):
				from fret_plot.ui.ui_hdf5view import hdf5_dataset_load
				try:
					self.hdf5_dataset_load.new_dataset.disconnect()
					self.hdf5_dataset_load.close()
					del self.hdf5_dataset_load
				except:
					pass

				try:
					self.hdf5_dataset_load = hdf5_dataset_load(filename=fname[0])
					self.hdf5_dataset_load.show()
					self.hdf5_dataset_load.new_dataset.connect(lambda x: self._load_hdf5(x,fname[0]))
					return False
				except:
					pass
				return False

			else:
				d = data_container(self)
				success = d.load_file(fname[0])
				if success:
					self._load(d)
					return True
				else:
					message = 'Could not load file: %s.'%(fname[0])
					QMessageBox.critical(None,'Could Not Load File',message)
					self.log(message,True)
					return False

	def _load_hdf5_dataset(self,filename,datasetname):
		import h5py as h
		with h.File(filename,'r') as f:
			dataset = f[datasetname][:]
		return self._load_hdf5(dataset,filename)

	def _load_hdf5(self,nd,filename):
		d = data_container(self)
		success = d.load(nd,filename)
		if success:
			self._load(d)
			return True
		else:
			return False

	def _load(self,d):
		## d should be a data_container

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


		self.docks['render'][1].spin_start.setMaximum(self.data.total_frames)
		self.docks['render'][1].spin_end.setMaximum(self.data.total_frames)
		self.docks['render'][1].spin_end.setValue(self.data.total_frames)

		self.plot.image.set_data(self.data.movie[self.data.current_frame])

		if self.docks['contrast'][1].flag_first_time:
			self.docks['contrast'][1].flag_first_time = False
			self.docks['contrast'][1].guess_contrast()
		else:
			self.data.image_contrast = np.array((float(self.docks['contrast'][1].le_floor.text()),float(self.docks['contrast'][1].le_ceiling.text())))

		self.plot.image.set_cmap(self.prefs['plot_colormap'])
		self.plot.draw()
		self.plot.canvas.draw()

		self.log('Loaded %s'%(self.data.filename),True)

		self.setWindowTitle('%s - %s'%(self.app_name,self.data.filename))
		self.prefs['movie_filename'] =self.data.filename

		self.docks['spotfind'][1].setup_sliders()
		self.docks['spotfind'][1].flush_old()
		self.docks['tag_viewer'][1].init_model()

	def log(self,line,timestamp = False):
		self._log.log(line,timestamp)
		self.set_status(line)

	def about(self):
		QMessageBox.about(None,'About %s'%(self.app_name),self.about_text)

	def set_status(self,message=""):
		self.statusbar.showMessage(message)
		self.app.processEvents()

	def dark_cal(self):
		if not self.data.movie is None:
			offset = self.prefs['calibrate_offset']
			gain = self.prefs['calibrate_gain']
			self.data.calibrate(gain,offset)
			self.log('Converted to electrons with %f,%f'%(gain,offset))

	def launch_psftool(self):
		from .ui_psf import psf_tool
		try:
			self.psf_tool.close()
			del self.psf_tool
		except:
			pass

		self.psf_tool = psf_tool(gui=self)
		self.psf_tool.show()


################################################################################
	def resizeEvent(self,event):
		if not self.signalsBlocked():
			s = self.size()
			sw = 0
			if not self.qd_prefs.isHidden() and not self.qd_prefs.isFloating():
				sw = self.qd_prefs.size().width()
			self.prefs['ui_width'] = s.width()-sw
			self.prefs['ui_height'] = s.height()
			super(vbscope_gui,self).resizeEvent(event)

	def open_log(self):
		self._open_ui(self._log)

	def _open_ui(self,ui):
		try:
			if not ui.isVisible():
				ui.setVisible(True)
			ui.raise_()
		except:
			ui.show()
		ui.showNormal()
		ui.activateWindow()


	def resize_prefs(self):
		w = self.prefs['ui_width']
		h = self.prefs['ui_height']
		self.blockSignals(True)
		if not self.qd_prefs.isHidden() and not self.qd_prefs.isFloating():
			sw = self.qd_prefs.size().width()
			self.resize(w+sw+4,h) ## ugh... +4 for dock handles
			if not self.centralWidget() == 0:
				self.centralWidget().resize(w,h)
		else:
			self.resize(w,h)
		self.blockSignals(False)

	def open_preferences(self):
		if self.qd_prefs.isHidden():
			self.qd_prefs.show()
			self.qd_prefs.raise_()
			self.prefs.le_filter.setFocus()

		else:
			self.qd_prefs.setHidden(True)
		self.resize_prefs()

	def ui_update(self):
		self.plot.toolbar.setStyleSheet('color:%s;background-color:%s;'%(self.prefs['ui_fontcolor'],self.prefs['ui_bgcolor']))
		self.plot.f.set_facecolor(self.prefs['ui_bgcolor'])
		self.plot.canvas.draw()

		for s in [self,self._log,self.prefs]:
			s.setStyleSheet('''
			color:%s;
			background-color:%s;
			font-size: %spx;
		'''%(self.prefs['ui_fontcolor'],self.prefs['ui_bgcolor'],self.prefs['ui_fontsize']))

		self.blockSignals(True)
		sw = 0
		if not self.qd_prefs.isHidden() and not self.qd_prefs.isFloating():
			sw = self.qd_prefs.size().width()
		self.resize(self.prefs['ui_width']+sw,self.prefs['ui_height'])
		self.blockSignals(False)
		self.setWindowTitle(self.app_name)

	def safe_close(self,event):
		reply = QMessageBox.question(self,"Quit?","Are you sure you want to quit?",QMessageBox.Yes | QMessageBox.No)
		if reply == QMessageBox.Yes:
			# import sys
			event.accept()
			## there's a weird double close that this avoids
			self.closeEvent = super(vbscope_gui,self).closeEvent
		else:
			event.ignore()

	def quick_close(self):
		self.closeEvent = super(vbscope_gui,self).closeEvent
		self.close()


def launch_scriptable(app=None):
	if app is None:
		app = QApplication([])
		app.setStyle('fusion')

	gui = vbscope_gui(app = app)
	return gui

def launch_gui():
	import os,sys
	gui = launch_scriptable()
	gui.show()
	try:
		__IPYTHON__
		return gui
	except:
		path = os.path.join(os.path.dirname(sys.modules[__name__].__file__), 'icon.png')
		gui.app.setWindowIcon(QIcon(path))

		sys.exit(gui.app.exec_())
