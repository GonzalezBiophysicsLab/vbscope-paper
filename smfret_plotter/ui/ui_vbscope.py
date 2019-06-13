from PyQt5.QtWidgets import QApplication,QDesktopWidget,QDockWidget, QAction
from PyQt5.QtCore import Qt

import sys
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')

from smfret_plotter.ui import movie_viewer
from src import docks
from smfret_plotter.containers import data_container
from smfret_plotter.containers import popout_plot_container
from src import plots
from .ui_ensemble_plot import gui_ensemble_plot

default_prefs = {
	'render_title':'vbscope',
	'render_artist':'vbscope',
	'plot_fontsize':12,

	'channels_colors':['green','red','blue','purple'],
	'channels_wavelengths':np.array((570.,680.,488.,800.))
}

class vbscope_gui(movie_viewer):
	def __init__(self,app=None):
		self.app = app

		super(vbscope_gui,self).__init__(self.app)

		self.prefs.add_dictionary(default_prefs)

		self.setup_vbscope_docks()
		self.setup_vbscope_menus()
		self.setup_vbscope_plots()

		self.app_name = 'vbscope'
		self.about_text = "From the Gonzalez Lab (Columbia University).\n\nPrinciple authors: JH,CKT,RLG.\nMany thanks to the entire lab for their input."

		self.prefs['ui_width']  = (QDesktopWidget().availableGeometry(self).size() * 0.7).width()
		self.prefs['ui_height'] = (QDesktopWidget().availableGeometry(self).size() * 0.7).height()
		self.move(0,0)

		self.ui_update()
		self.show()

	def setup_vbscope_docks(self):
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


	def setup_vbscope_menus(self):
		## Add menu items
		menu_analysis = self.menubar.addMenu('Analysis')
		m = ['spotfind', 'background', 'transform', 'extract']
		for mm in m:
			menu_analysis.addAction(self.docks[mm][0].toggleViewAction())
		self.menu_movie.addAction(self.docks['tag_viewer'][0].toggleViewAction())

	def setup_vbscope_plots(self):
		menu_plot = self.menubar.addMenu('Plots')
		plt_region = QAction('Region plot',self)
		plt_region.triggered.connect(self.plot_region)

		for a in [plt_region]:
			menu_plot.addAction(a)

		self.popout_plots = {
			'plot_region':None
		}

		plots_ensemble = QAction('Ensemble Plots',self)
		plots_ensemble.triggered.connect(self.show_ensemble_plot)
		menu_plot.addAction(plots_ensemble)
		self.ui_update()

	def raise_plot(self,plot_handle,plot_name_str="Plot",nplots=1,callback=None,dprefs=None):
		try:
			ph = self.popout_plots[plot_handle]
			if not ph.isVisible():
				ph.setVisible(True)
			ph.raise_()
		except:
			self.popout_plots[plot_handle] = popout_plot_container(nplots,self)
			self.popout_plots[plot_handle].setWindowTitle(plot_name_str)
			if not dprefs is None:
				self.popout_plots[plot_handle].ui.prefs.add_dictionary(dprefs)
			if not callback is None:
				self.popout_plots[plot_handle].ui.setcallback(callback)
			self.popout_plots[plot_handle].show()
			self.popout_plots[plot_handle].ui.clf()

	def plot_region(self):
		pp = plots.region.default_prefs
		self.raise_plot('plot_region', 'Region Plot', 1, lambda: plots.region.plot(self), pp)
		plots.region.plot(self)


	def load(self,fname=None):
		success = super(vbscope_gui, self).load(fname)
		if success:
			self.docks['spotfind'][1].setup_sliders()
			self.docks['spotfind'][1].flush_old()
			self.docks['tag_viewer'][1].init_model()

	def show_ensemble_plot(self):
		try:
			if not self.ui_ensemble_plot.isVisible():
				self.ui_ensemble_plot.setVisible(True)
			self.ui_ensemble_plot.raise_()
		except:
			self.ui_ensemble_plot = gui_ensemble_plot(self)
			self.ui_ensemble_plot.setWindowTitle('Ensemble Plots')
			self.ui_ensemble_plot.show()


def launch_vbscope(scriptable=True):
	'''
	Launch the main window as a standalone GUI (ie without vbscope analyze movies), or for scripting.
	----------------------
	Example:
	from vbscope import launch
	----------------------
	'''

	import sys
	app = QApplication([])
	app.setStyle('fusion')
	g = vbscope_gui(app)

	if scriptable:
		return g
	else:
		sys.exit(app.exec_())
