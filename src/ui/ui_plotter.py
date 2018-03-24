from PyQt5.QtWidgets import QMainWindow, QWidget, QSizePolicy, QVBoxLayout, QShortcut, QSlider, QHBoxLayout, QPushButton, QFileDialog, QCheckBox,QApplication, QAction,QLineEdit,QLabel,QGridLayout, QInputDialog, QDockWidget, QMessageBox, QTabWidget, QListWidget, QAbstractItemView
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator, QKeySequence

import multiprocessing as mp

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


from . import ui_general
from .ui_batch_loader import gui_batch_loader
from ..containers import traj_plot_container, traj_container, popout_plot_container
from .. import plots

default_prefs = {

	'channel_colors':['green','red','blue','purple'],

	'tau':0.1,
	'bleedthrough':np.array(((0.,0.05,0.,0.),(0.,0.,0.,0.),(0.,0.,0.,0.),(0.,0.,0.,0.))).flatten(),

	'ncpu':mp.cpu_count(),

	'downsample':1,
	'snr_threshold':.5,
	'pb_length':10,
    #
	# 'plotter_floor':0.2,
	# 'plotter_nbins_contour':20,
	# 'plotter_smoothx':0.5,
	# 'plotter_smoothy':0.5,
	# 'plotter_timeshift':0.0,
	# 'plotter_floorcolor':'lightgoldenrodyellow',
	# 'plotter_cmap':'rainbow',
	'plotter_min_fret':-.5,
	'plotter_max_fret':1.5,
	'plotter_wiener_smooth':0,
	# 'plotter_nbins_fret':41,
	# 'plotter_min_time':0,
	# 'plotter_max_time':100,
	# 'plotter_nbins_time':100,
	# 'plotter_2d_syncpreframes':10,
	# 'plotter_2d_normalizecolumn':False,

	'convert_flag':False,
	'convert_c_lambda':[11.64,12.75],
	'convert_em_gain':300,
	'convert_offset':0,

	'photobleaching_flag':True,
	'synchronize_start_flag':False,

	'hmm_nrestarts':4,
	'hmm_sigmasmooth':False,
	'hmm_binding_expt':False,
	'hmm_bound_dynamics':False
}

## GUI for plotting 2D smFRET trajectories
class plotter_gui(ui_general.gui):
	def __init__(self,data,gui):
		self.gui = gui
		self.data = traj_container(self)

		self.initialize_ui()

		super(plotter_gui,self).__init__(self.gui.app,self._main_widget)
		self._prefs.combine_prefs(default_prefs)

		self.initialize_connections()

		if not data is None:
			self.initialize_data(data.astype('double'))
			self.plot.index = 0
			self.plot.initialize_plots()

			self.initialize_sliders()
			self.update_display_traces()
			self.log("Loaded %d traces from vbscope gui"%(self.data.d.shape[0]),True)

		self.initialize_plot_docks()

		self.app_name = 'vbscope plotter'
		self.about_text = 'plotting tool'
		self.setWindowTitle(self.app_name)

		self.ui_update()
		self.show()

	def initialize_ui(self):
		## Initialize Plots
		self.ncolors = self.gui.data.ncolors
		self.plot = traj_plot_container(self)

		## Initialize trajectory slider
		self.slider_select = QSlider(Qt.Horizontal)
		sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
		self.slider_select.setSizePolicy(sizePolicy)

		qw = QWidget()
		hbox = QHBoxLayout()
		self.label_current = QLabel("0 / 0 - 0")

		hbox.addWidget(self.slider_select)
		hbox.addWidget(self.label_current)
		qw.setLayout(hbox)

		### Initialize line edit boxes for y-min and y-max intensity values
		self.plot.yminmax = np.array((0.,0.))

		scalewidget = QWidget()
		g = QGridLayout()
		g.addWidget(QLabel('Min'),0,0)
		g.addWidget(QLabel('Max'),0,1)
		self.le_min = QLineEdit()
		self.le_max = QLineEdit()
		g.addWidget(self.le_min,1,0)
		g.addWidget(self.le_max,1,1)

		for ll in [self.le_min,self.le_max]:
			ll.setValidator(QDoubleValidator(-1e300,1e300,100))
			ll.setText(str(0.))
		scalewidget.setLayout(g)

		## Put all of the widgets together
		self._main_widget = QWidget()
		self.layout = QVBoxLayout()
		self.layout.addWidget(self.plot.canvas)
		self.layout.addWidget(self.plot.toolbar)
		# self.layout.addWidget(self.slider_select)
		self.layout.addWidget(qw)
		self.layout.addWidget(scalewidget)
		self._main_widget.setLayout(self.layout)

		plt.close(self.plot.f)


	def initialize_connections(self):
		## Initialize Menu options
		self.initialize_menubar()

		# ## Connect everything for interaction
		self.initialize_shortcuts()
		self.initialize_sliders()
		for le in [self.le_min,self.le_max]:
			le.editingFinished.connect(self.plot.update_minmax)
		self.slider_select.valueChanged.connect(self.callback_sliders)
		self.plot.f.canvas.mpl_connect('button_press_event', self.callback_mouseclick)

	## Set the trajectory selection limits
	def initialize_sliders(self):
		self.slider_select.setMinimum(0)
		mm = 0
		if not self.data.d is None:
			mm = self.data.d.shape[0]-1
		self.slider_select.setMaximum(mm)
		self.slider_select.setValue(self.plot.index)

	## Reset everything with the new trajectories loaded in with this function
	def initialize_data(self,data,sort=True):
		self.data.d = data

		## Guess at good y-limits for the plot
		self.plot.yminmax = np.percentile(self.data.d.flatten()[np.isfinite(self.data.d.flatten())],[.1,99.9])
		self.le_min.setText(str(self.plot.yminmax[0]))
		self.le_max.setText(str(self.plot.yminmax[1]))

		## Sort trajectories based on anti-correlation (most is first, least is last)
		if sort:
			self.data.cross_corr_order()

		## Calculate/set photobleaching, initialize class list
		self.data.update_fret()
		self.data.pre_list = np.zeros(self.data.d.shape[0],dtype='i')
		self.data.pb_list = self.data.pre_list.copy() + self.data.d.shape[2]
		self.data.class_list = np.zeros(self.data.d.shape[0])

		self.data.deadprob = np.zeros(self.data.pre_list.size)


	## Setup the menu items at the top
	def initialize_menubar(self):
		## turn off old load action
		fo = self.menubar.actions()[0].menu().actions()[0]
		fo.setVisible(False)
		fo.setShortcut(QKeySequence())

# 		### Load
		menu_load = self.menubar.addMenu('Load')
		load_load_traces = QAction('Load Traces', self, shortcut='Ctrl+O')
		load_load_traces.triggered.connect(lambda event: self.load_traces())

		load_load_classes = QAction('Load Classes', self, shortcut='Ctrl+P')
		load_load_classes.triggered.connect(lambda event: self.load_classes())

		load_load_hmm = QAction('Load HMM', self)
		load_load_hmm.triggered.connect(lambda event: self.load_hmm())

		load_batch = QAction('Batch Load',self, shortcut='Ctrl+B')
		load_batch.triggered.connect(self.show_batch_load)

		for f in [load_load_traces,load_load_classes,load_load_hmm,load_batch]:
			menu_load.addAction(f)
			#
		### save
		menu_save = self.menubar.addMenu('Export')

		export_traces = QAction('Save Traces', self, shortcut='Ctrl+S')
		export_traces.triggered.connect(lambda event: self.export_traces())

		export_processed_traces = QAction('Save Processed Traces', self)
		export_processed_traces.triggered.connect(lambda event: self.export_processed_traces())

		export_classes = QAction('Save Classes', self, shortcut='Ctrl+D')
		export_classes.triggered.connect(lambda event: self.export_classes())

		for f in [export_traces,export_classes,export_processed_traces]:
			menu_save.addAction(f)

		### tools
		menu_tools = self.menubar.addMenu('Tools')
		tools_cullpb = QAction('Cull PB', self)
		tools_cullpb.triggered.connect(self.data.cull_pb)
		tools_cullmin = QAction('Cull minimums', self)
		tools_cullmin.triggered.connect(lambda event: self.data.cull_min())
		tools_cullmax = QAction('Cull maximums', self)
		tools_cullmax.triggered.connect(lambda event: self.data.cull_max())
		tools_cullphotons = QAction('Cull Photons',self)
		tools_cullphotons.triggered.connect(lambda event: self.data.cull_photons())
		tools_step = QAction('Photobleach - Step',self)
		tools_step.triggered.connect(self.data.photobleach_step)
		tools_var = QAction('Remove all PB - Var',self)
		tools_var.setCheckable(True)
		tools_var.setChecked(False)
		self.pb_remove_check = tools_var
		tools_remove = QAction('Remove From Beginning',self)
		tools_remove.triggered.connect(lambda event: self.data.remove_beginning())
		tools_dead = QAction('Remove Dead Traces',self)
		tools_dead.triggered.connect(self.data.remove_dead)
		tools_hmm = QAction('HMM',self)
		tools_hmm.triggered.connect(lambda event: self.data.run_hmm())

		for f in [tools_cullpb,tools_cullmin,tools_cullmax,tools_cullphotons,tools_step,tools_var,tools_remove,tools_dead,tools_hmm]:
			menu_tools.addAction(f)
#
		### plots
		menu_plots = self.menubar.addMenu('Plots')

		plots_1d = QAction('1D Histogram', self)
		plots_1d.triggered.connect(self.plot_hist1d)
		plots_2d = QAction('2D Histogram', self)
		plots_2d.triggered.connect(self.plot_hist2d)
		plots_tdp = QAction('Transition Density Plot', self)
		plots_tdp.triggered.connect(self.plot_tdp)
		plots_tranM = QAction('Transition Matrix Plot', self)
		plots_tranM.triggered.connect(self.plot_tranM)
		plots_intensities = QAction('Intensities Plot', self)
		plots_intensities.triggered.connect(self.plot_intensities)

		for f in [plots_1d,plots_2d,plots_tdp, plots_tranM, plots_intensities]:
			menu_plots.addAction(f)

		### classes
		menu_classes = self.menubar.addMenu("Classes")

		self.action_classes = []
		for i in range(10):
			m = QAction(str(i),self)
			m.setCheckable(True)
			m.setChecked(True)
			self.action_classes.append(m)

		toggle = QAction("Toggle all",self)
		toggle.triggered.connect(self.classes_toggle_all)

		counts = QAction("Class Counts",self)
		counts.triggered.connect(self.classes_show_counts)

		separator = QAction(self)
		separator.setSeparator(True)

		menu_classes.addAction(counts)
		menu_classes.addAction(toggle)
		menu_classes.addAction(separator)
		for m in self.action_classes:
			menu_classes.addAction(m)


################################################################################

	## Helper function to setup keyboard shortcuts
	def initialize_shortcuts(self):
		## Helper function to setup keyboard shortcuts
		def _make_shortcut(key,fxn):
			qs = QShortcut(self)
			qs.setKey(key)
			qs.activated.connect(fxn)

		_make_shortcut(Qt.Key_Left,lambda : self.callback_keypress('left'))
		_make_shortcut(Qt.Key_Right,lambda : self.callback_keypress('right'))
		_make_shortcut(Qt.Key_H,lambda : self.callback_keypress('h'))
		_make_shortcut(Qt.Key_1,lambda : self.callback_keypress(1))
		_make_shortcut(Qt.Key_2,lambda : self.callback_keypress(2))
		_make_shortcut(Qt.Key_3,lambda : self.callback_keypress(3))
		_make_shortcut(Qt.Key_4,lambda : self.callback_keypress(4))
		_make_shortcut(Qt.Key_5,lambda : self.callback_keypress(5))
		_make_shortcut(Qt.Key_6,lambda : self.callback_keypress(6))
		_make_shortcut(Qt.Key_7,lambda : self.callback_keypress(7))
		_make_shortcut(Qt.Key_8,lambda : self.callback_keypress(8))
		_make_shortcut(Qt.Key_9,lambda : self.callback_keypress(9))
		_make_shortcut(Qt.Key_0,lambda : self.callback_keypress(0))
		_make_shortcut(Qt.Key_R,lambda : self.callback_keypress('r'))

	def initialize_plot_docks(self):
		self.popout_plots = {
			'plot_hist1d':None,
			'plot_hist2d':None,
			'plot_tdp':None,
			'plot_tranM':None,
			'plot_intensities':None
		}

	def raise_plot(self,plot_handle,plot_name_str="Plot",nplots_x=1, nplots_y=1,callback=None,dprefs=None):
		try:
			ph = self.popout_plots[plot_handle]
			if not ph.isVisible():
				ph.setVisible(True)
			ph.raise_()
		except:
			self.popout_plots[plot_handle] = popout_plot_container(nplots_x, nplots_y,self)
			self.popout_plots[plot_handle].setWindowTitle(plot_name_str)
			if not dprefs is None:
				self.popout_plots[plot_handle].ui._prefs.combine_prefs(dprefs)
			if not callback is None:
				self.popout_plots[plot_handle].ui.setcallback(callback)
			self.popout_plots[plot_handle].resize(int(self.popout_plots[plot_handle].ui.prefs['fig_width']*self.plot.f.get_dpi()/self.plot.canvas.devicePixelRatio())+200,int(self.popout_plots[plot_handle].ui.prefs['fig_height']*self.plot.f.get_dpi()/self.plot.canvas.devicePixelRatio())+125)
			self.popout_plots[plot_handle].show()
			self.popout_plots[plot_handle].ui.clf()

	def plot_hist1d(self):
		self.raise_plot('plot_hist1d', '1D Histogram', 1,1, lambda: plots.hist_1d.plot(self), plots.hist_1d.default_prefs)
		plots.hist_1d.plot(self)

	def plot_hist2d(self):
		self.raise_plot('plot_hist2d', '2D Histogram', 1,1, lambda: plots.hist_2d.plot(self), plots.hist_2d.default_prefs)
		plots.hist_2d.plot(self)

	def plot_tdp(self):
		self.raise_plot('plot_tdp', 'Transition Density Plot', 1,1, lambda: plots.tdp.plot(self), plots.tdp.default_prefs)
		plots.tdp.plot(self)

	def plot_tranM(self):
		self.raise_plot('plot_tranM', 'Transition Matrix Plot', 1,1, lambda: plots.tranM.plot(self), plots.tranM.default_prefs)
		plots.tranM.plot(self)

	def plot_intensities(self):
		self.raise_plot('plot_intensities', 'Cumulative Intensities', 1,3, lambda: plots.intensities.plot(self), plots.intensities.default_prefs)
		plots.intensities.plot(self)

	## Callback function for keyboard presses
	def callback_keypress(self,kk):
		if kk == 'right':
			self.plot.index += 1
		elif kk == 'left':
			self.plot.index -= 1
		if self.plot.index < 0:
			self.plot.index = 0
		elif self.plot.index >= self.data.d.shape[0]:
			self.plot.index = self.data.d.shape[0]-1
		self.slider_select.setValue(self.plot.index)

		if kk == 'r':
			self.data.pre_list[self.plot.index] = 0
			self.data.pb_list[self.plot.index] = self.data.d.shape[2]-1

		for i in range(10):
			if i == kk:
				self.data.class_list[self.plot.index] = kk
				self.plot.update_plots()
				break

		try:
			self.update_display_traces()
			self.gui.app.processEvents()
		except:
			pass

		self.plot.update_plots()



	## callback function for changing the trajectory using the slider
	def callback_sliders(self,v):
		self.plot.index = v
		self.update_display_traces()
		self.gui.app.processEvents()
		self.plot.update_plots()

	## Handler for mouse clicks in main plots
	def callback_mouseclick(self,event):
		if ((event.inaxes == self.plot.a[0][0] or event.inaxes == self.plot.a[1][0])) and not self.data.d is None:
			## Right click - set photobleaching point
			if event.button == 3 and self.plot.toolbar._active is None:
				self.data.pb_list[self.plot.index] = int(np.round(event.xdata/self.prefs['tau']))
				self.data.safe_hmm()
				self.plot.update_plots()
			## Left click - set pre-truncation point
			if event.button == 1 and self.plot.toolbar._active is None:
				self.data.pre_list[self.plot.index] = int(np.round(event.xdata/self.prefs['tau']))
				self.data.safe_hmm()
				self.plot.update_plots()
			## Middle click - reset pre and post points to calculated values
			if event.button == 2 and self.plot.toolbar._active is None:
				if self.ncolors == 2:
					from ..supporting.photobleaching import get_point_pbtime
					self.data.pre_list[self.plot.index] = 0
					if self.prefs['photobleaching_flag'] is True:
						qq = self.data.d[self.plot.index].sum(0)
					else:
						qq = self.data.d[self.plot.index,1]
					self.data.pb_list[self.plot.index] = get_point_pbtime(qq)
					self.data.safe_hmm()
					self.plot.update_plots()

################################################################################

	def update_display_traces(self):
		self.label_current.setText("{n1:0{w}} / {n2:0{w}} - {n3:1d} - {n4:2f}".format(n1 = self.plot.index + 0, n2 = self.data.d.shape[0] - 1, n3 = int(self.data.class_list[self.plot.index]), n4=self.data.deadprob[self.plot.index], w =int(np.floor(np.log10(self.data.d.shape[0]))+1)))

	def classes_get_checked(self):
		checked = np.zeros(self.data.d.shape[0],dtype='bool')
		for i in range(10):
			if self.action_classes[i].isChecked():
				checked[np.nonzero(self.data.class_list == i)] = True
		return checked

	def classes_show_counts(self):
		if not self.data.d is None:
			report = "Class\tCounts\n"
			for i in range(10):
				c = (self.data.class_list == i).sum()
				report += str(i)+'\t%d\n'%(c)
			self.log(report)

	def classes_toggle_all(self):
		for m in self.action_classes:
			m.setChecked(not m.isChecked())

################################################################################

	def show_batch_load(self):
		try:
			if not self.ui_batch.isVisible():
				self.ui_batch.setVisible(True)
			self.ui_batch.raise_()
		except:
			self.ui_batch = gui_batch_loader(self)
			self.ui_batch.setWindowTitle('Batch Load')
			self.ui_batch.show()

################################################################################

	def load_batch(self,ltraj,lclass,ncolors = None):
		if ncolors is None:
			ncolors,success = QInputDialog.getInt(self,"Number of Color Channels","Number of Color Channels",value=2,min=1)
		else:
			success = True
		if success:
			ds = []
			cs = []
			for i in range(len(ltraj)):
				try:
					# d = np.loadtxt(ltraj[i],delimiter=',').T
					d = self.quicksafe_load(ltraj[i]).T
					dd = np.array([d[j::ncolors] for j in range(ncolors)])
					d = np.moveaxis(dd,1,0)

					if not lclass[i] is None:
						# c = np.loadtxt(lclass[i],delimiter=',').astype('i')
						c = self.quicksafe_load(lclass[i]).astype('i')
					else:
						c = np.zeros((d.shape[0],1+2*ncolors),dtype='i')
						c[:,2::2] = d.shape[2]
					ds.append(d)
					cs.append(c)
				except:
					self.log("Could not load:\n\t%s\n\t%s"%(ltraj[i],lclass[i]),True)

			if len(cs) > 0:
				cc = np.concatenate(cs,axis=0)
				maxlength = np.max([dd.shape[2] for dd in ds]).astype('i')
				for i in range(len(ds)):
					dsi = ds[i]
					dtemp = np.zeros((dsi.shape[0],dsi.shape[1],maxlength))
					dtemp[:,:,:dsi.shape[2]] = dsi
					ds[i] = dtemp
				dd = np.concatenate(ds,axis=0)

				self.ncolors = ncolors
				self.initialize_data(dd,sort=False)
				self.plot.index = 0
				self.plot.initialize_plots()
				self.initialize_sliders()

				self.data.class_list = cc[:,0]
				self.data.pre_list = cc[:,1::2].max(1)
				self.data.pb_list = cc[:,2::2].min(1)
				self.plot.update_plots()

				try:
					self.ui_batch.close()
				except:
					pass
				self.update_display_traces()
				msg = "Loaded %d pairs of files containing a total of %d trajectories."%(len(ds),dd.shape[0])
				# QMessageBox.information(self,"Batch Load",msg)
				self.log(msg,True)
			else:
				msg = "Could not find any traces in these files"
				# QMessageBox.critical(self,"Batch Load",msg)
				self.log(msg,True)

	def load_hmm(self, fname=None):
		if not self.data.d is None:
			if fname is None:
				fname,_ = QFileDialog.getOpenFileName(self,'Choose HMM result file','./')
			if fname != "":
				try:
				# if 1:
					import cPickle as pickle
					f = open(fname,'r')
					self.data.hmm_result = pickle.load(f)
					f.close()
					self.plot.initialize_hmm_plot()
					self.plot.update_plots()
					self.log("Loaded HMM result from %s"%(fname),True)
				except:
					self.log("Failed to load HMM result",True)

	## Try to load trajectories from a vbscope style file (commas)
	def load_traces(self,filename=None,checked=False,ncolors=2):
		if filename is None:
			fname,_ = QFileDialog.getOpenFileName(self,'Choose file to load traces','./')#,filter='TIF File (*.tif *.TIF)')
			if fname is u"":
				return
		else:
			fname = filename

		if not fname is u"":
			success = False
			if filename is None:
				ncolors,success2 = QInputDialog.getInt(self,"Number of Color Channels","Number of Color Channels",value=2,min=1)
			else:
				success2 = True

			try:
				# d = np.loadtxt(fname,delimiter=',').T
				d = self.quicksafe_load(fname).T
				dd = np.array([d[i::ncolors] for i in range(ncolors)])
				d = np.moveaxis(dd,1,0)
				success = True
			except:
				pass

			if success and success2:
				self.ncolors = ncolors
				self.initialize_data(d,sort=False)
				self.plot.index = 0
				if filename is None:
					self.plot.initialize_plots()
					self.initialize_sliders()
					self.update_display_traces()
				self.log("Loaded traces from %s"%(fname),True)
				return
		self.log("Could not load %s"%(fname),True)

	## Try to load classes/bleach times from a vbscope style file (commas) (Nx5)
	def load_classes(self,filename=None,checked=False):
		if filename is None:
			fname,_ = QFileDialog.getOpenFileName(self,'Choose file to load classes','./')#,filter='TIF File (*.tif *.TIF)')
			if fname is u"":
				return
		else:
			fname = filename

		if not fname is "":
			success = False
			try:
				# d = np.loadtxt(fname,delimiter=',').astype('i')
				d = self.quicksafe_load(fname).astype('i')
				if d.shape[0] == self.data.d.shape[0]:
					success = True
			except:
				pass

			if success:
				self.data.class_list = d[:,0]
				self.data.pre_list = d[:,1::2].max(1)
				self.data.pb_list = d[:,2::2].min(1)
				if filename is None:
					self.plot.update_plots()
					self.update_display_traces()
				self.log("Loaded classes from %s"%(fname),True)
				return
		self.log("Could not load %s"%(fname),True)

################################################################################


	## Save raw donor-acceptor trajectories (bleedthrough corrected) in vbscope format (commas)
	def export_processed_traces(self, format = None, oname = None):
		n = self.ncolors
		if not self.data.d is None:
			dd = np.copy(self.data.d)

			# Bleedthrough
			bts = self.prefs['bleedthrough'].reshape((4,4))
			for i in range(self.ncolors):
				for j in range(self.ncolors):
					dd[:,j] -= bts[i,j]*dd[:,i]

			checked = self.classes_get_checked()

			combos = ['2D','1D','SMD']
			if format is None:
				c,success = QInputDialog.getItem(self,"Format","Choose Format of Exported Data",combos,editable=False)
			else:
				c = format
				success = True
			if success:
					try:
						if c == '1D':
							identities = np.array([])
							j = 0
							ds = [np.array([]) for _ in range(n)]
							for i in range(dd.shape[0]):
								if checked[i] == 1:
									pre = self.data.pre_list[i]
									post = self.data.pb_list[i]
									identities = np.append(identities, np.repeat(j,post-pre))
									for k in range(self.ncolors):
										ds[k] = np.append(ds[k],dd[i,k,pre:post])
									j += 1
							q = np.vstack((identities,ds)).T
							if oname is None:
								oname = QFileDialog.getSaveFileName(self, 'Export Processed Traces', '_processedtraces.dat','*.dat')
							else:
								oname = [oname]
							if oname[0] != "":
								np.savetxt(oname[0],q,delimiter=',')
								self.log("Exported as 1D",True)

						elif c == '2D': # 2D
							# Photobleaching
							for i in range(dd.shape[0]):
								pre = self.data.pre_list[i]
								post = self.data.pb_list[i]
								dd[i,:,:post-pre] = dd[i,:,pre:post]
								dd[i,:,post-pre:] = np.nan

							# Classes
							dd = dd[checked]

							q = np.zeros((dd.shape[0]*n,dd.shape[2]))
							for i in range(n):
								q[i::n] = dd[:,i]
							q = q.T
							if oname is None:
								oname = QFileDialog.getSaveFileName(self, 'Export Processed Traces', '_processedtraces.dat','*.dat')
							else:
								oname = [oname]
							if oname[0] != "":
								np.savetxt(oname[0],q,delimiter=',')
								self.log("Exported as 2D",True)

						if c == 'SMD' and n == 2:
							from time import ctime
							from hashlib import md5
							teatime = ctime()
							hashed= md5(teatime + str(np.random.rand())).hexdigest()
							ttype = np.array([[ (np.array([u'vbscope'],dtype='<U42'),)]],dtype=[('session', 'O')])
							spoofid = np.array([hashed], dtype='<U38')
							fake_attr = np.array('none')
							cols = np.array([[np.array([u'fret'],dtype='<U4'),np.array([u'donor'], dtype='<U5'), np.array([u'acceptor'],dtype='<U8')]], dtype='O')
							q = {'type':ttype,'id':spoofid,'attr':fake_attr,'columns':cols}
							dt = np.dtype([('id', 'O'), ('index', 'O'), ('values', 'O'), ('attr', 'O')])
							data = []
							for i in range(dd.shape[0]):
								if checked[i] == 1:
									pre = self.data.pre_list[i]
									post = self.data.pb_list[i]
									hashed= md5(teatime + str(np.random.rand())).hexdigest()
									spoofid = np.array([hashed], dtype='<U38')
									o = np.array((spoofid,np.arange(post-pre),np.vstack((np.zeros(post-pre),dd[i,:,pre:post])).T,fake_attr),dtype=dt)
									data.append(o)
							q['data'] = np.hstack(data)
							if oname is None:
								oname = QFileDialog.getSaveFileName(self, 'Export Processed Traces', '_processedtraces.mat','*.mat')
							else:
								oname = [oname]
							if oname[0] != "":
								from scipy.io.matlab import savemat
								savemat(oname[0],q)
								self.log("Exported as SMD",True)

					except:
						QMessageBox.critical(self,'Export Processed Traces','There was a problem trying to export the processed traces')

	## Save raw donor-acceptor trajectories (bleedthrough corrected) in vbscope format (commas)
	def export_traces(self,oname = None):
		n = self.ncolors
		if not self.data.d is None:
			dd = self.data.d.copy()
			# if n == 2:
			# 	dd[:,1] -= self.gui.prefs['bleedthrough']*dd[:,0]

			checked = self.classes_get_checked()
			dd = dd[checked]
			q = np.zeros((dd.shape[0]*n,dd.shape[2]))
			for i in range(n):
				q[i::n] = dd[:,i]

			if oname is None:
				oname = QFileDialog.getSaveFileName(self, 'Export Traces', '_traces.dat','*.dat')
			else:
				oname = [oname]
			if oname[0] != "":
				try:
					np.savetxt(oname[0],q.T,delimiter=',')
					self.log("Exported traces",True)
				except:
					msg = 'There was a problem trying to export the traces'
					QMessageBox.critical(self,'Export Traces',msg)

	## Save classes/bleach times in the vbscope format (Nx5) (with commas)
	def export_classes(self,oname=None):
		n = self.ncolors #number of colors
		if not self.data.d is None:
			q = np.zeros((self.data.d.shape[0],1+2*n),dtype='int')
			q[:,0] = self.data.class_list
			for i in range(n):
				q[:,1+2*i] = self.data.pre_list
				q[:,1+2*i+1] = self.data.pb_list
			checked = self.classes_get_checked()
			q = q[checked]

			if oname is None:
				oname = QFileDialog.getSaveFileName(self, 'Export Classes/Cuts', '_classes.dat','*.dat')
			else:
				oname = [oname]
			if oname[0] != "":
				try:
					np.savetxt(oname[0],q.astype('i'),delimiter=',')
					self.log("Exported classes",True)
				except:
					msg = 'There was a problem trying to export the classes/cuts'
					QMessageBox.critical(self,'Export Classes',msg)
					self.log(msg,True)


def launch_plotter(scriptable=True):
	'''
	Launch the main window as a standalone GUI (ie without vbscope analyze movies), or for scripting.
	----------------------
	Example:
	from plotter import launch

	g = launch()
	g.ui.load_traces_m('RF2(271)_mutArfA_1uM_combined_traces.dat')
	g.ui.load_classes_m('RF2(271)_mutArfA_1uM_combined_classes.dat')
	g.ui.hist2d()
	popplot = g.ui.docks['plots_2D'][1]
	popplot.f.savefig('test.pdf')
	----------------------
	'''

	class _fake(object):
		pass

	import sys
	app = QApplication([])
	app.setStyle('fusion')
	gui = _fake()
	gui.app = app
	gui.data = _fake()
	gui.data.ncolors = 2
	g = plotter_gui(None,gui)
	if scriptable:
		return g
	else:
		sys.exit(gui.app.exec_())
