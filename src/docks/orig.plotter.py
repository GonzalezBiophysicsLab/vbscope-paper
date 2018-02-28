from PyQt5.QtWidgets import QMainWindow, QWidget, QSizePolicy, QVBoxLayout, QShortcut, QSlider, QHBoxLayout, QPushButton, QFileDialog, QCheckBox,QApplication, QAction,QLineEdit,QLabel,QGridLayout, QInputDialog, QDockWidget, QMessageBox, QTabWidget, QListWidget, QAbstractItemView
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

import numpy as np

default_prefs = {
}
# from prefs import dock_prefs

## GUI for plotting 2D smFRET trajectories
class ui_plotter(QMainWindow):
	def __init__(self,data,spots,bg=None,parent=None,):
		super(QMainWindow,self).__init__(parent)

		## If there's no vbscope movie GUI, then still need to have access to the preferences
		if not parent is None:
			self.gui = parent.gui
			self.ncolors = self.gui.data.ncolors
		else:
			self.gui = fake()
			# from prefs import default as dfprefs
			# self.gui.prefs = dfprefs
			self.ncolors = 2

		## Make the plotter widget
		self.ui = plotter(data,self)
		self.setCentralWidget(self.ui)
		self.setWindowTitle('vbscope plot')
		self.show()

	def closeEvent(self,event):
		try:
			self.parent().activateWindow()
			self.parent().raise_()
			self.parent().setFocus()
		except:
			pass
		print 'goodbye'

## The QWidget that is embedded in the main window class `ui_plotter`
class plotter(QWidget):
	def __init__(self,data,parent=None):
		super(QWidget,self).__init__(parent=parent)
		self.default_prefs = default_prefs
		self.gui = parent.gui
		layout = QVBoxLayout()

		self.ncolors = parent.ncolors

		## Docks
		self.docks = {}
		# self.docks['prefs'] = [QDockWidget('Preferences',parent),dock_prefs(self)]
		# self.docks['prefs'][0].setWidget(self.docks['prefs'][1])
		# self.docks['prefs'][0].setAllowedAreas(Qt.NoDockWidgetArea)
		# parent.addDockWidget(Qt.BottomDockWidgetArea, self.docks['prefs'][0])
		# self.docks['prefs'][0].setFloating(True)
		# self.docks['prefs'][0].close()

		self.docks['plots_1D'] = [QDockWidget('1D Histogram',parent),mpl_plot()]
		self.docks['plots_1D'][0].setWidget(self.docks['plots_1D'][1])
		self.docks['plots_1D'][0].setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
		parent.addDockWidget(Qt.RightDockWidgetArea, self.docks['plots_1D'][0])
		self.docks['plots_1D'][0].close()
		self.docks['plots_2D'] = [QDockWidget('2D Histogram',parent),mpl_plot()]
		self.docks['plots_2D'][0].setWidget(self.docks['plots_2D'][1])
		self.docks['plots_2D'][0].setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
		parent.addDockWidget(Qt.RightDockWidgetArea, self.docks['plots_2D'][0])
		self.docks['plots_2D'][0].close()
		self.docks['plots_TDP'] = [QDockWidget('Transition Density',parent),mpl_plot()]
		self.docks['plots_TDP'][0].setWidget(self.docks['plots_TDP'][1])
		self.docks['plots_TDP'][0].setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
		parent.addDockWidget(Qt.RightDockWidgetArea, self.docks['plots_TDP'][0])
		self.docks['plots_TDP'][0].close()
		self.docks['plots_surv'] = [QDockWidget('Survival Plot',parent),mpl_plot()]
		self.docks['plots_surv'][0].setWidget(self.docks['plots_surv'][1])
		self.docks['plots_surv'][0].setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
		parent.addDockWidget(Qt.RightDockWidgetArea, self.docks['plots_surv'][0])
		self.docks['plots_surv'][0].close()

		self.parent().tabifyDockWidget(self.docks['plots_1D'][0],self.docks['plots_2D'][0])
		self.parent().tabifyDockWidget(self.docks['plots_2D'][0],self.docks['plots_TDP'][0])
		self.parent().tabifyDockWidget(self.docks['plots_TDP'][0],self.docks['plots_surv'][0])
		self.parent().setTabPosition(Qt.AllDockWidgetAreas, QTabWidget.North)

		## Initialize Plots
		self.f,self.a = plt.subplots(2,2,gridspec_kw={'width_ratios':[6,1]},figsize=(6.5,4))
		self.canvas = FigureCanvas(self.f)
		self.f.set_dpi(self.f.get_dpi()/self.canvas.devicePixelRatio())
		sizePolicy = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
		self.canvas.setSizePolicy(sizePolicy)
		self.toolbar = NavigationToolbar(self.canvas,None)

		## Initialize trajectory slider
		self.slider_select = QSlider(Qt.Horizontal)
		sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
		self.slider_select.setSizePolicy(sizePolicy)

		### Initialize line edit boxes for y-min and y-max intensity values
		scalewidget = QWidget()
		g = QGridLayout()
		g.addWidget(QLabel('Min'),0,0)
		g.addWidget(QLabel('Max'),0,1)
		self.le_min = QLineEdit()
		self.le_max = QLineEdit()
		self.yminmax = np.array((0.,0.))
		g.addWidget(self.le_min,1,0)
		g.addWidget(self.le_max,1,1)
		for ll in [self.le_min,self.le_max]:
			ll.setValidator(QDoubleValidator(-1e300,1e300,100))
			ll.editingFinished.connect(self.update_minmax)
			ll.setText(str(0.))
		scalewidget.setLayout(g)

		## Put all of the widgets together
		layout.addWidget(self.canvas)
		layout.addWidget(self.toolbar)
		layout.addWidget(self.slider_select)
		layout.addWidget(scalewidget)
		self.setLayout(layout)

		## Initialize Menu options
		self.setup_menubar()

		## Initialize beginnning data (or lack thereof)
		self.index = 0
		self.d = data
		self.hmm_result = None
		if not self.d is None:
			self.initialize_data(data.astype('double'))
			self.initialize_plots()
			self.update()

		## Connect everything for interaction
		self.init_shortcuts()
		self.initialize_sliders()
		self.slider_select.valueChanged.connect(self.slide_switch)
		self.f.canvas.mpl_connect('button_press_event', self.mouse_click)
		plt.close(self.f)

	## Read in y-min and y-max values, then update the plot
	def update_minmax(self):
		self.yminmax = np.array((float(self.le_min.text()),float(self.le_max.text())))
		self.update()
		for le in [self.le_min,self.le_max]:
			try:
				le.clearFocus()
			except:
				pass

	## Set the trajectory selection limits
	def initialize_sliders(self):
		self.slider_select.setMinimum(0)
		mm = 0
		if not self.d is None:
			mm = self.d.shape[0]-1
		self.slider_select.setMaximum(mm)
		self.slider_select.setValue(self.index)

	## Plot initial data to set aesthetics
	def initialize_plots(self):
		## clear everything
		[[aaa.cla() for aaa in aa] for aa in self.a]

		lw=.75 / self.canvas.devicePixelRatio()
		pb=.2

		for i in range(self.ncolors):
			## plot pre-truncated, kept, and post-truncated trajectory (Intensities)
			color = self.gui.prefs['channel_colors'][i]
			self.a[0][0].plot(np.random.rand(self.d.shape[0]),color=color,ls=':',alpha=pb,lw=lw)
			self.a[0][0].plot(np.random.rand(self.d.shape[0]),color=color,alpha=.8,lw=lw)
			self.a[0][0].plot(np.random.rand(self.d.shape[0]),color=color,ls=':',alpha=pb,lw=lw)

			## Plot histograms of intensities
			self.a[0][1].plot(np.random.rand(100),color=color,alpha=.8,lw=1./ self.canvas.devicePixelRatio())

		## plot pre-truncated, kept, and post-truncated trajectory (E_{FRET})
		for i in range(1,self.ncolors):
			if self.ncolors == 2:
				color = 'blue'
			else:
				color = self.gui.prefs['channel_colors'][i]
			self.a[1][0].plot(np.random.rand(self.d.shape[0]),color=color,ls=':',alpha=pb,lw=lw)
			self.a[1][0].plot(np.random.rand(self.d.shape[0]),color=color,alpha=.8,lw=lw)
			self.a[1][0].plot(np.random.rand(self.d.shape[0]),color=color,ls=':',alpha=pb,lw=lw)
			## Plot histograms of rel. intensities
			self.a[1][1].plot(np.random.rand(100),color=color,alpha=.8,lw=1./ self.canvas.devicePixelRatio())

		## Make it so that certain plots zoom together
		self.a[0][0].get_shared_y_axes().join(self.a[0][0],self.a[0][1])
		self.a[1][0].get_shared_y_axes().join(self.a[1][0],self.a[1][1])
		self.a[0][0].get_shared_x_axes().join(self.a[0][0],self.a[1][0])
		self.a[0][1].get_shared_x_axes().join(self.a[0][1],self.a[1][1])

		## Set the ticks/labels so that they look nice
		for aa in self.a:
			for aaa in aa:
				for asp in ['top','bottom','left','right']:
					aaa.spines[asp].set_linewidth(1.0/self.canvas.devicePixelRatio())
				aaa.tick_params(labelsize=12./self.canvas.devicePixelRatio(),axis='both',direction='in',width=1.0/self.canvas.devicePixelRatio(),length=2./self.canvas.devicePixelRatio())
				aaa.tick_params(axis='both',which='major',length=4./self.canvas.devicePixelRatio())

		plt.setp(self.a[0][0].get_xticklabels(), visible=False)
		for aa in [self.a[0][1],self.a[1][1]]:
			aa.yaxis.tick_right()
			plt.setp(aa.get_yticklabels(),visible=False)
			plt.setp(aa.get_xticklabels(),visible=False)
			# aa.tick_params(axis='x', which='both',length=0)
		self.a[0][1].tick_params(axis='x', which='both',length=0)
		self.a[0][0].tick_params(axis='x', which='both',length=0)
		self.a[0][1].tick_params(axis='y',which='both',direction='in')
		self.a[1][1].tick_params(axis='y',which='both',direction='in')

		## Redraw everything
		self.update()
		self.label_axes()
		self.f.tight_layout()
		offset1 = .08
		offset2 = 0.02
		offset3 = 0.14
		self.f.subplots_adjust(left=offset3,right=1.-offset2,top=1.-offset1,bottom=offset3,hspace=.03,wspace=0.015)
		self.f.canvas.draw()

	## Reset everything with the new trajectories loaded in with this function
	def initialize_data(self,data,sort=True):
		self.d = data

		## Guess at good y-limits for the plot
		self.yminmax = np.percentile(self.d.flatten(),[.1,99.9])
		self.le_min.setText(str(self.yminmax[0]))
		self.le_max.setText(str(self.yminmax[1]))

		## Sort trajectories based on anti-correlation (most is first, least is last)
		if sort:
			order = self.cross_corr_order()
			self.d = self.d[order]

		## Calculate/set photobleaching, initialize class list
		self.update_fret()
		# self.fret = pb.remove_pb_all(self.fret)
		# cut = np.isnan(self.fret)
		# self.d[:,0][cut] = np.nan
		# self.d[:,0][cut] = np.nan
		self.pre_list = np.zeros(self.d.shape[0],dtype='i')
		self.pb_list = self.pre_list.copy() + self.d.shape[2]
		self.class_list = np.zeros(self.d.shape[0])

	def update_fret(self):
		q = np.copy(self.d)
		bts = self.gui.prefs['bleedthrough'].reshape((4,4))
		for i in range(self.ncolors):
			for j in range(self.ncolors):
				q[:,j] -= bts[i,j]*q[:,i]
		# C-1,N,T
		self.fret = np.array([q[:,i]/q.sum(1) for i in range(1,self.ncolors)])

	def get_sum_fluor(self):
		q = np.copy(self.d)
		bts = self.gui.prefs['bleedthrough'].reshape((4,4))
		for i in range(self.ncolors):
			for j in range(self.ncolors):
				q[:,j] -= bts[i,j]*q[:,i]
		return q.sum(1)


	## Setup the menu items at the top
	def setup_menubar(self):
		self.menubar = self.parent().menuBar()
		self.menubar.setNativeMenuBar(False)

		### File
		menu_file = self.menubar.addMenu('File')

		file_load_traces = QAction('Load Traces', self, shortcut='Ctrl+O')
		file_load_traces.triggered.connect(self.load_traces)

		file_load_classes = QAction('Load Classes', self, shortcut='Ctrl+P')
		file_load_classes.triggered.connect(self.load_classes)

		file_load_hmm = QAction('Load HMM', self)
		file_load_hmm.triggered.connect(self.load_hmm)

		file_batch = QAction('Batch Load',self, shortcut='Ctrl+B')
		file_batch.triggered.connect(self.batch_load)

		file_about = QAction('About',self)
		file_about.triggered.connect(self.about)

		file_exit = QAction('Exit', self, shortcut='Ctrl+Q')
		file_exit.triggered.connect(self.parent().close)

		for f in [file_load_traces,file_load_classes,file_load_hmm,file_batch,self.docks['prefs'][0].toggleViewAction(),file_about,file_exit]:
			menu_file.addAction(f)

		### save
		menu_save = self.menubar.addMenu('Export')

		export_traces = QAction('Save Traces', self, shortcut='Ctrl+S')
		export_traces.triggered.connect(self.export_traces)

		export_processed_traces = QAction('Save Processed Traces', self)
		export_processed_traces.triggered.connect(self.export_processed_traces)

		export_classes = QAction('Save Classes', self, shortcut='Ctrl+D')
		export_classes.triggered.connect(self.export_classes)

		for f in [export_traces,export_classes,export_processed_traces]:
			menu_save.addAction(f)

		### tools
		menu_tools = self.menubar.addMenu('Tools')

		tools_cull = QAction('Cull SNR', self)
		tools_cull.triggered.connect(self.cull_snr)
		tools_cullpb = QAction('Cull PB', self)
		tools_cullpb.triggered.connect(self.cull_pb)
		# tools_cullempty = QAction('Cull Empty',self)
		# tools_cullempty.triggered.connect(self.cull_empty)
		tools_cullphotons = QAction('Cull Photons',self)
		tools_cullphotons.triggered.connect(self.cull_photons)
		tools_step = QAction('Photobleach - Step',self)
		tools_step.triggered.connect(self.photobleach_step)
		tools_var = QAction('Remove all PB - Var',self)
		tools_var.setCheckable(True)
		tools_var.setChecked(False)
		self.pb_remove_check = tools_var
		tools_remove = QAction('Remove From Beginning',self)
		tools_remove.triggered.connect(self.remove_beginning)
		tools_hmm = QAction('HMM',self)
		tools_hmm.triggered.connect(self.run_hmm)


		for f in [tools_cull,tools_cullpb,tools_cullphotons,tools_step,tools_var,tools_remove,tools_hmm]:
			menu_tools.addAction(f)

		### plots
		menu_plots = self.menubar.addMenu('Plots')

		plots_1d = QAction('1D Histogram', self)
		plots_1d.triggered.connect(self.hist1d)
		plots_2d = QAction('2D Histogram', self)
		plots_2d.triggered.connect(self.hist2d)
		plots_tdp = QAction('Transition Density Plot', self)
		plots_tdp.triggered.connect(self.tdplot)

		for f in [plots_1d,plots_2d,plots_tdp]:
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
		toggle.triggered.connect(self.toggle_all_classes)

		counts = QAction("Class Counts",self)
		counts.triggered.connect(self.show_class_counts)

		separator = QAction(self)
		separator.setSeparator(True)

		menu_classes.addAction(counts)
		menu_classes.addAction(toggle)
		menu_classes.addAction(separator)
		for m in self.action_classes:
			menu_classes.addAction(m)

	def about(self):
		from prefs import last_update_date
		QMessageBox.about(None,'About vbscope','Version: %s\n\nFrom the Gonzalez Lab (Columbia University).\n\nPrinciple authors: JH,CKT,RLG.\nMany thanks to the entire lab for their input.'%(last_update_date))


	def remove_beginning(self):
		if not self.d is None:
			self.safe_hmm()
			nd,success = QInputDialog.getInt(self,"Remove Datapoints","Number of datapoints to remove starting from the beginning of the movie")
			if success and nd > 1 and nd < self.d.shape[2]:
				self.index = 0
				self.initialize_data(self.d[:,:,nd:])
				self.initialize_plots()
				self.initialize_sliders()

	def photobleach_step(self):
		if not self.d is None and self.ncolors == 2:
			self.safe_hmm()
			from supporting.photobleaching import pb_ensemble
			q = np.copy(self.d)
			q[:,1] -= self.gui.prefs['bleedthrough'][1]*q[:,0]
			# l1 = calc_pb_time(self.fret,self.gui.prefs['pb_length_cutoff'])
			if self.gui.prefs['photobleaching_flag'] is True:
				qq = q[:,0] + q[:,1]
				print 'hey'
			else:
				qq = q[:,1]
				print 'ho'
			l2 = pb_ensemble(qq)[1]
			# self.pb_list = np.array([np.min((l1[i],l2[i])) for i in range(l1.size)])
			self.pb_list = l2
			self.update()

	def run_hmm(self):
		from supporting import simul_vbem_hmm as hmm

		if not self.d is None and self.ncolors == 2:
			if self.gui.prefs['hmm_binding_expt'] == 'True':
				nstates = 2
				success = True
			else:
				nstates,success = QInputDialog.getInt(self,"Number of HMM States","Number of HMM States",min=2)
			if success and nstates > 1:
				self.update_fret()
				y = []
				checked = self.get_checked()
				ran = []
				if self.gui.prefs['hmm_binding_expt'] == 'True':
					z = self.get_sum_fluor()
				for i in range(self.fret.shape[1]):
					if checked[i]:
						if self.gui.prefs['hmm_binding_expt'] == 'True':
							yy = z[i,self.pre_list[i]:self.pb_list[i]]
						else:
							yy = self.fret[0,i,self.pre_list[i]:self.pb_list[i]]
							yy[np.isnan(yy)] = -1.
							yy[yy < -1.] = -1.
							yy[yy > 2] = 2.
						if yy.size > 5:
							y.append(yy)
							ran.append(i)
				nrestarts = self.gui.prefs['hmm_nrestarts']
				priors = [hmm.initialize_priors(y,nstates,flag_vbfret=False,flag_custom=True) for _ in range(nrestarts)]
				if self.gui.prefs['hmm_binding_expt'] == 'True':
					for iii in range(nrestarts):
						priors[iii][0] = np.array((0,1000.)) ## m
						priors[iii][1] = np.ones(2) ## beta



				if self.gui.prefs['hmm_sigmasmooth'] == "True":
					sigma_smooth = 0.5
				else:
					sigma_smooth = False
				result,lbs = hmm.hmm(y,nstates,priors,nrestarts,sigma_smooth)
				ppi = np.sum([result.gamma[i].sum(0) for i in range(len(result.gamma))],axis=0)
				ppi /= ppi.sum()

				print lbs
				print '\nHMM - k = %d, iter= %d, lowerbound=%f'%(nstates,result.iterations,result.lowerbound)
				print '    f:',ppi
				print '    m:',result.m
				print 'm_sig',1./np.sqrt(result.beta)
				print '  sig:',(result.b/result.a)**.5
				rates = -np.log(1.-result.Astar)/self.gui.prefs['tau']
				for i in range(rates.shape[0]):
					rates[i,i] = 0.
				print '  k:'
				print rates
				self.hmm_result = result
				self.hmm_result.ran = ran
				if len(self.a[1,0].lines) < 4:
					self.a[1,0].plot(np.random.rand(100),np.random.rand(100),color='k',lw=1.,alpha=.8)
				self.update()

				########### EXPORT

				# q = np.zeros((self.d.shape[0],self.d.shape[2])) + np.nan
				# j = 0
				# for i in range(self.d.shape[0]):
				# 	if self.hmm_result.ran.count(i)>0:
				# 		pbtime = int(self.pb_list[i])
				# 		pretime = int(self.pre_list[i])
				# 		q[i,pretime:pbtime] = self.hmm_result.m[self.hmm_result.viterbi[j]]
				# 		j += 1
				# oname = QFileDialog.getSaveFileName(self, 'Export Viterbi Paths', '_viterbi.dat','*.dat')
				# if oname[0] != "":
				# 	try:
				# 		np.savetxt(oname[0],q.T,delimiter=',')
				# 	except:
				# 		QMessageBox.critical(self,'Export Traces','There was a problem trying to export the viterbi paths')

				import cPickle as pickle
				oname = QFileDialog.getSaveFileName(self, 'Export HMM results', '_HMM.dat','*.dat')
				if oname[0] != "":
					try:
						f = open(oname[0],'w')
						pickle.dump(self.hmm_result, f)
						f.close()
					except:
						QMessageBox.critical(self,'Export Traces','There was a problem trying to export the HMM results')

				if self.gui.prefs['hmm_binding_expt'] == 'True':
					oname = QFileDialog.getSaveFileName(self, 'Save Chopped Traces', '_chopped.dat','*.dat')
					if oname[0] == "":
						return

					## N,C,T
					out = None

					from scipy.ndimage import label as ndilabel
					from scipy.ndimage import find_objects as ndifind

					## If it has an HMM
					for j in range(len(self.hmm_result.ran)): ## hmm index
						i = self.hmm_result.ran[j] ## trace index

						v = self.hmm_result.viterbi[j]

						pre = int(self.pre_list[i])
						post = int(self.pb_list[i])
						q = self.d[i,:,pre:post]

						labels,numlabels = ndilabel(v)
						slices = ndifind(labels)
						if len(slices)>0:
							for ss in slices:
								ss = ss[0]
								tmp = np.zeros((1,self.d.shape[1],self.d.shape[2]))
								tmp[0,:,:ss.stop-ss.start] = self.d[i,:,pre+ss.start:pre+ss.stop]
								tmp_cl = np.array((0,0,ss.stop-ss.start,0,ss.stop-ss.start))[None,:]
								if out is None:
									out = tmp.copy()
									classes = tmp_cl.copy()
								else:
									out = np.append(out,tmp,axis=0)
									classes = np.append(classes,tmp_cl,axis=0)

					q = np.zeros((out.shape[0]*out.shape[1],out.shape[2]))
					for i in range(out.shape[1]):
						q[i::out.shape[1]] = out[:,i]
					np.savetxt(oname[0],q.T,delimiter=',')
					np.savetxt(oname[0][:-4]+"_classes.dat",classes.astype('i'),delimiter=',')




	def load_hmm(self):
		if not self.d is None:
			fname = QFileDialog.getOpenFileName(self,'Choose HMM result file','./')
			if fname[0] != "":
				try:
				# if 1:
					import cPickle as pickle
					f = open(fname[0],'r')
					self.hmm_result = pickle.load(f)
					f.close()
					if len(self.a[1,0].lines) < 4:
						self.a[1,0].plot(np.random.rand(100),np.random.rand(100),color='k',lw=1.,alpha=.8)
					self.update()
				except:
					print "failed"

	def show_class_counts(self):
		if not self.d is None:
			report = "Class\tCounts\n"
			for i in range(10):
				c = (self.class_list == i).sum()
				report += str(i)+'\t%d\n'%(c)
			print report

	def toggle_all_classes(self):
		for m in self.action_classes:
			m.setChecked(not m.isChecked())

	def safe_hmm(self):
		if not self.hmm_result is None:
			self.hmm_result = None
			self.safe_hmm()

	## Remove trajectories with SNR less than threshold
	def cull_snr(self):
		if not self.d is None and self.ncolors == 2:
			snr_threshold = self.gui.prefs['snr_threshold']
			self.safe_hmm()
			# snr = np.zeros(self.d.shape[0])
			# for i in range(snr.size):
			# 	q = self.d[i].sum(0)
			# 	snr[i] = q.mean()/np.std(q)
			from supporting.photobleaching import pb_snr
			snr = pb_snr(self.d.sum(1))
			cut = snr > snr_threshold
			print "kept %d out of %d = %f"%(cut.sum(),snr.size,cut.sum()/float(snr.size))

			if self.docks['plots_surv'][0].isHidden():
				self.docks['plots_surv'][0].show()
			self.docks['plots_surv'][0].raise_()
			popplot = self.docks['plots_surv'][1]
			popplot.ax.cla()

			x = np.linspace(np.min((0.,snr.min())),20,10000)
			surv = np.array([(snr > x[i]).sum()/float(snr.size) for i in range(x.size)])
			# x = np.linspace(0.,15,131)
			# y = np.zeros_like(x)
			# for i in range(y.size):
				# y[i] = (snr > x[i]).sum()
			# y/=y[0]
			popplot.ax.plot(x,surv)
			popplot.ax.axvline(snr_threshold,color='r')
			popplot.ax.set_ylim(0,1)
			popplot.ax.set_ylabel('Survival Probability')
			popplot.ax.set_xlabel('SNR')
			popplot.f.tight_layout()
			popplot.f.canvas.draw()

			d = self.d[cut]
			self.index = 0
			self.initialize_data(d)
			self.initialize_plots()
			self.initialize_sliders()

	## Remove trajectories with number of kept-frames < threshold
	def cull_pb(self):
		if not self.d is None and self.ncolors == 2:
			self.safe_hmm()
			pbt = self.pb_list.copy()
			pret = self.pre_list.copy()
			dt = pbt-pret
			cut = dt > self.gui.prefs['pb_length']
			print "kept %d out of %d = %f"%(cut.sum(),pbt.size,cut.sum()/float(pbt.size))
			d = self.d[cut]
			pbt = pbt[cut]
			pret = pret[cut]
			print d.shape,pbt.shape,pret.shape
			self.index = 0
			self.initialize_data(d,sort=False)
			self.pb_list = pbt
			self.pre_list = pret
			self.initialize_plots()
			self.initialize_sliders()


	# def cull_empty(self):
	# 	if not self.d is None:
	# 		combos = ['%d'%(i) for i in range(self.ncolors)]
	# 		combos.append('0+1')
	# 		c,success1 = QInputDialog.getItem(self,"Color","Choose Color channel",combos,editable=False)
	# 		if success1:
	# 			from supporting.photobleaching import model_comparison_signal
	# 			self.safe_hmm()
	# 			y = np.zeros((self.d.shape[0],self.d.shape[2]))
	# 			for ind in range(self.d.shape[0]):
	# 				intensities = self.d[ind].copy()
	# 				bts = self.gui.prefs['bleedthrough'].reshape((4,4))
	# 				for i in range(self.ncolors):
	# 					for j in range(self.ncolors):
	# 						intensities[j] -= bts[i,j]*intensities[i]
	#
	# 				for i in range(self.ncolors):
	# 					intensities[i] = self.gui.prefs['convert_c_lambda'][i]/self.gui.prefs['convert_em_gain']*intensities[i]
	# 				cc = c.split('+')
	#
	# 				for ccc in cc:
	# 					y[ind] += intensities[int(ccc)]
	# 			keep = model_comparison_signal(y)
	#
	# 			x = np.linspace(0,1,1000)
	# 			surv = np.array([(keep > x[i]).sum()/float(keep.size) for i in range(x.size)])
	# 			plt.figure()
	# 			plt.plot(x,surv)
	# 			plt.ylabel('Survival Probability')
	# 			plt.xlabel('Probability Not Empty')
	# 			plt.xlim(0,1)
	# 			plt.ylim(0,1)
	# 			plt.show()
	# 			cutp,success2 = QInputDialog.getDouble(self,"P Cutoff","Choose probability cutoff (keep above)",value=.95,min=0.,max=1.,decimals=10)
	# 			if success2:
	# 				keep = keep > cutp
	# 				d = self.d[keep]
	# 				self.index = 0
	# 				self.initialize_data(d)
	# 				self.initialize_plots()
	# 				self.initialize_sliders()

	def cull_photons(self):
		if not self.d is None:
			combos = ['%d'%(i) for i in range(self.ncolors)]
			combos.append('0+1')
			c,success1 = QInputDialog.getItem(self,"Color","Choose Color channel",combos,editable=False)
			if success1:
				self.safe_hmm()
				keep = np.zeros(self.d.shape[0],dtype='bool')
				y = np.zeros(self.d.shape[0])
				for ind in range(self.d.shape[0]):
					intensities = self.d[ind].copy()
					bts = self.gui.prefs['bleedthrough'].reshape((4,4))
					for i in range(self.ncolors):
						for j in range(self.ncolors):
							intensities[j] -= bts[i,j]*intensities[i]

					for i in range(self.ncolors):
						intensities[i] = self.gui.prefs['convert_c_lambda'][i]/self.gui.prefs['convert_em_gain']*intensities[i]
					cc = c.split('+')
					for ccc in cc:
						y[ind] += intensities[int(ccc)].sum()
				# x = np.logspace(0,np.log10(y.max()),1000)
				x = np.linspace(y.min(),y.max(),10000)
				surv = np.array([(y > x[i]).sum()/float(y.size) for i in range(x.size)])

				if self.docks['plots_surv'][0].isHidden():
					self.docks['plots_surv'][0].show()
				self.docks['plots_surv'][0].raise_()
				popplot = self.docks['plots_surv'][1]
				popplot.ax.cla()

				# plt.semilogx(x,surv)
				popplot.ax.plot(x,surv)
				popplot.ax.set_ylim(0,1)
				popplot.ax.set_ylabel('Survival Probability')
				popplot.ax.set_xlabel('Total Number of Photons')
				popplot.f.tight_layout()
				popplot.f.canvas.draw()
				threshold,success2 = QInputDialog.getDouble(self,"Photon Cutoff","Total number of photons required to keep a trajectory",value=1000.,min=0.,max=1e10,decimals=10)
 				if success2:
					keep = y > threshold
					d = self.d[keep]
					self.index = 0
					self.initialize_data(d)
					self.initialize_plots()
					self.initialize_sliders()

	## Save raw donor-acceptor trajectories (bleedthrough corrected) in vbscope format (commas)
	def export_processed_traces(self):
		n = self.ncolors
		if not self.d is None:
			dd = np.copy(self.d)

			# Bleedthrough
			bts = self.gui.prefs['bleedthrough'].reshape((4,4))
			for i in range(self.ncolors):
				for j in range(self.ncolors):
					dd[:,j] -= bts[i,j]*dd[:,i]

			checked = self.get_checked()

			combos = ['2D','1D','SMD']
			c,success = QInputDialog.getItem(self,"Format","Choose Format of Exported Data",combos,editable=False)
			if success:
					try:
						if c == '1D':
							identities = np.array([])
							j = 0
							ds = [np.array([]) for _ in range(n)]
							for i in range(dd.shape[0]):
								if checked[i] == 1:
									pre = self.pre_list[i]
									post = self.pb_list[i]
									identities = np.append(identities, np.repeat(j,post-pre))
									for k in range(self.ncolors):
										ds[k] = np.append(ds[k],dd[i,k,pre:post])
									j += 1
							q = np.vstack((identities,ds)).T
							oname = QFileDialog.getSaveFileName(self, 'Export Processed Traces', '_processedtraces.dat','*.dat')
							if oname[0] != "":
								np.savetxt(oname[0],q,delimiter=',')

						elif c == '2D': # 2D
							# Photobleaching
							for i in range(dd.shape[0]):
								pre = self.pre_list[i]
								post = self.pb_list[i]
								dd[i,:,:post-pre] = dd[i,:,pre:post]
								dd[i,:,post-pre:] = np.nan

							# Classes
							dd = dd[checked]

							q = np.zeros((dd.shape[0]*n,dd.shape[2]))
							for i in range(n):
								q[i::n] = dd[:,i]
							q = q.T
							oname = QFileDialog.getSaveFileName(self, 'Export Processed Traces', '_processedtraces.dat','*.dat')
							if oname[0] != "":
								np.savetxt(oname[0],q,delimiter=',')

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
									pre = self.pre_list[i]
									post = self.pb_list[i]
									hashed= md5(teatime + str(np.random.rand())).hexdigest()
									spoofid = np.array([hashed], dtype='<U38')
									o = np.array((spoofid,np.arange(post-pre),np.vstack((np.zeros(post-pre),dd[i,:,pre:post])).T,fake_attr),dtype=dt)
									data.append(o)
							q['data'] = np.hstack(data)
							oname = QFileDialog.getSaveFileName(self, 'Export Processed Traces', '_processedtraces.mat','*.mat')
							if oname[0] != "":
								from scipy.io.matlab import savemat
								savemat(oname[0],q)

					except:
						QMessageBox.critical(self,'Export Processed Traces','There was a problem trying to export the processed traces')

	## Save raw donor-acceptor trajectories (bleedthrough corrected) in vbscope format (commas)
	def export_traces(self):
		n = self.ncolors
		if not self.d is None:
			dd = self.d.copy()
			# if n == 2:
			# 	dd[:,1] -= self.gui.prefs['bleedthrough']*dd[:,0]

			checked = self.get_checked()
			dd = dd[checked]
			q = np.zeros((dd.shape[0]*n,dd.shape[2]))
			for i in range(n):
				q[i::n] = dd[:,i]

			oname = QFileDialog.getSaveFileName(self, 'Export Traces', '_traces.dat','*.dat')
			if oname[0] != "":
				try:
					np.savetxt(oname[0],q.T,delimiter=',')
				except:
					QMessageBox.critical(self,'Export Traces','There was a problem trying to export the traces')

	## Save classes/bleach times in the vbscope format (Nx5) (with commas)
	def export_classes(self):
		n = self.ncolors #number of colors
		if not self.d is None:
			q = np.zeros((self.d.shape[0],1+2*n),dtype='int')
			q[:,0] = self.class_list
			for i in range(n):
				q[:,1+2*i] = self.pre_list
				q[:,1+2*i+1] = self.pb_list
			checked = self.get_checked()
			q = q[checked]

			oname = QFileDialog.getSaveFileName(self, 'Export Classes/Cuts', '_classes.dat','*.dat')
			if oname[0] != "":
				try:
					np.savetxt(oname[0],q.astype('i'),delimiter=',')
				except:
					QMessageBox.critical(self,'Export Classes','There was a problem trying to export the classes/cuts')

	## Try to load trajectories from a vbscope style file (commas)
	def load_traces(self):
		fname = QFileDialog.getOpenFileName(self,'Choose file to load traces','./')#,filter='TIF File (*.tif *.TIF)')
		if fname[0] != "":
			success = False
			ncolors,suc2 = QInputDialog.getInt(self,"Number of Color Channels","Number of Color Channels",value=2,min=1)
			try:

				d = np.loadtxt(fname[0],delimiter=',').T
				dd = np.array([d[i::ncolors] for i in range(ncolors)])
				d = np.moveaxis(dd,1,0)
				success = True
			except:
				print "could not load %s"%(fname[0])

			if success and suc2:
				self.ncolors = ncolors
				self.initialize_data(d,sort=False)
				self.index = 0
				self.initialize_plots()
				self.initialize_sliders()

	def load_traces_m(self, fname):

		ncolors = 2
		d = np.loadtxt(fname,delimiter=',').T
		dd = np.array([d[i::ncolors] for i in range(ncolors)])
		d = np.moveaxis(dd,1,0)

		self.ncolors = ncolors
		self.initialize_data(d,sort=False)
		self.index = 0
		# self.initialize_plots()
		# self.initialize_sliders()

	## Try to load classes/bleach times from a vbscope style file (commas) (Nx5)
	def load_classes(self):
		fname = QFileDialog.getOpenFileName(self,'Choose file to load classes','./')#,filter='TIF File (*.tif *.TIF)')
		if fname[0] != "":
			success = False
			try:
				d = np.loadtxt(fname[0],delimiter=',').astype('i')
				if d.shape[0] == self.d.shape[0]:
					success = True
			except:
				print "could not load %s"%(fname[0])

			if success:
				self.class_list = d[:,0]
				self.pre_list = d[:,1::2].max(1)
				self.pb_list = d[:,2::2].min(1)
				self.update()

	def load_classes_m(self, fname):

		d = np.loadtxt(fname,delimiter=',').astype('i')
		if d.shape[0] == self.d.shape[0]:
			self.class_list = d[:,0]
			self.pre_list = d[:,1::2].max(1)
			self.pb_list = d[:,2::2].min(1)
			# self.update()

	## Handler for mouse clicks in main plots
	def mouse_click(self,event):
		if (event.inaxes == self.a[0][0] or event.inaxes == self.a[1][0]) and not self.d is None:
			## Right click - set photobleaching point
			if event.button == 3 and self.toolbar._active is None:
				self.pb_list[self.index] = int(np.round(event.xdata/self.gui.prefs['tau']))
				self.safe_hmm()
				self.update()
			## Left click - set pre-truncation point
			if event.button == 1 and self.toolbar._active is None:
				self.pre_list[self.index] = int(np.round(event.xdata/self.gui.prefs['tau']))
				self.safe_hmm()
				self.update()
			## Middle click - reset pre and post points to calculated values
			if event.button == 2 and self.toolbar._active is None:
				if self.ncolors == 2:
					from supporting.photobleaching import get_point_pbtime
					self.pre_list[self.index] = 0
					if self.gui.prefs['photobleaching_flag'] is True:
						qq = self.d[self.index].sum(0)
					else:
						qq = self.d[self.index,1]
					self.pb_list[self.index] = get_point_pbtime(qq)
					self.safe_hmm()
					self.update()

	## Add axis labels to plots
	def label_axes(self):
		fs = 12./self.canvas.devicePixelRatio()

		self.a[0][0].set_ylabel(r'Intensity (a.u.)',fontsize=fs,va='top')
		if self.ncolors == 2:
			self.a[1][0].set_ylabel(r'E$_{\rm{FRET}}$',fontsize=fs,va='top')
		else:
			self.a[1][0].set_ylabel(r'Relative Intensity',fontsize=fs,va='top')
		self.a[1][0].set_xlabel(r'Time (s)',fontsize=fs)
		self.a[1][1].set_xlabel(r'Probability',fontsize=fs)

	## callback function for changing the trajectory using the slider
	def slide_switch(self,v):
		self.index = v
		self.update()

	## Helper function to setup keyboard shortcuts
	def init_shortcuts(self):
		self.make_shortcut(Qt.Key_Left,lambda : self.key('left'))
		self.make_shortcut(Qt.Key_Right,lambda : self.key('right'))
		self.make_shortcut(Qt.Key_H,lambda : self.key('h'))
		self.make_shortcut(Qt.Key_1,lambda : self.key(1))
		self.make_shortcut(Qt.Key_2,lambda : self.key(2))
		self.make_shortcut(Qt.Key_3,lambda : self.key(3))
		self.make_shortcut(Qt.Key_4,lambda : self.key(4))
		self.make_shortcut(Qt.Key_5,lambda : self.key(5))
		self.make_shortcut(Qt.Key_6,lambda : self.key(6))
		self.make_shortcut(Qt.Key_7,lambda : self.key(7))
		self.make_shortcut(Qt.Key_8,lambda : self.key(8))
		self.make_shortcut(Qt.Key_9,lambda : self.key(9))
		self.make_shortcut(Qt.Key_0,lambda : self.key(0))

	## Helper function to setup keyboard shortcuts
	def make_shortcut(self,key,fxn):
		qs = QShortcut(self)
		qs.setKey(key)
		qs.activated.connect(fxn)

	## Callback function for keyboard presses
	def key(self,kk):
		if kk == 'right':
			self.index += 1
		elif kk == 'left':
			self.index -= 1
		if self.index < 0:
			self.index = 0
		elif self.index >= self.d.shape[0]:
			self.index = self.d.shape[0]-1
		self.slider_select.setValue(self.index)

		# if kk == 'h':
		# 	self.ensemble_hist()
		for i in range(10):
			if i == kk:
				self.class_list[self.index] = kk
				self.update()

		try:
			self.gui.app.processEvents()
		except:
			pass
		# self.update()

	def get_plot_data(self):
		# f = self.fret
		self.update_fret()
		fpb = self.fret.copy()
		for j in range(self.ncolors-1):
			for i in range(fpb.shape[1]):
				fpb[j][i,:self.pre_list[i]] = np.nan
				fpb[j][i,self.pb_list[i]:] = np.nan
				# if self.gui.prefs['synchronize_start_flag'] != 'True':
				# 	fpb[j][i,:self.gui.prefs['plotter_min_time']] = np.nan
				# 	fpb[j][i,self.gui.prefs['plotter_max_time']:] = np.nan

		checked = self.get_checked()
		fpb = fpb[:,checked]

		if self.pb_remove_check.isChecked() and self.ncolors == 2:
			from supporting.photobleaching import remove_pb_all
			fpb[0] = remove_pb_all(fpb[0])
		return fpb

	def get_viterbi_data(self):
		if not self.hmm_result is None:
			v = np.empty_like(self.fret[0]) + np.nan
			for i in range(v.shape[0]):
				if self.hmm_result.ran.count(i) > 0:
					ii = self.hmm_result.ran.index(i)
					v[i,self.pre_list[i]:self.pb_list[i]] = self.hmm_result.viterbi[ii]

			# if self.gui.prefs['synchronize_start_flag'] != 'True':
			# 	v[i,:self.gui.prefs['plotter_min_time']] = np.nan
			# 	v[i,self.gui.prefs['plotter_max_time']:] = np.nan

			checked = self.get_checked()
			v = v[checked]
			return v
		else:
			return None

	def get_checked(self):
		checked = np.zeros(self.d.shape[0],dtype='bool')
		for i in range(10):
			if self.action_classes[i].isChecked():
				checked[np.nonzero(self.class_list == i)] = True
		return checked

	## Plot the 1D Histogram of the ensemble
	def hist1d(self):
		if self.docks['plots_1D'][0].isHidden():
			self.docks['plots_1D'][0].show()
		self.docks['plots_1D'][0].raise_()
		popplot = self.docks['plots_1D'][1]
		popplot.ax.cla()

		if self.ncolors == 2:
			fpb = self.get_plot_data()[0]

			# plt.hist(f.flatten(),bins=181,range=(-.4,1.4),histtype='stepfilled',alpha=.8,normed=True)
			popplot.ax.hist(fpb.flatten(),bins=self.gui.prefs['plotter_nbins_fret'],range=(self.gui.prefs['plotter_min_fret'],self.gui.prefs['plotter_max_fret']),histtype='stepfilled',alpha=.8,normed=True)

			if not self.hmm_result is None:
				r = self.hmm_result
				def norm(x,m,v):
					return 1./np.sqrt(2.*np.pi*v)*np.exp(-.5/v*(x-m)**2.)
				x = np.linspace(self.gui.prefs['plotter_min_fret'],self.gui.prefs['plotter_max_fret'],1001)
				ppi = np.sum([r.gamma[i].sum(0) for i in range(len(r.gamma))],axis=0)
				ppi /=ppi.sum()
				v = r.b/r.a
				tot = np.zeros_like(x)
				for i in range(r.m.size):
					y = ppi[i]*norm(x,r.m[i],v[i])
					tot += y
					popplot.ax.plot(x,y,color='k',lw=1,alpha=.8,ls='--')
				popplot.ax.plot(x,tot,color='k',lw=2,alpha=.8)

			popplot.ax.set_xlim(self.gui.prefs['plotter_min_fret'],self.gui.prefs['plotter_max_fret'])
			popplot.ax.set_xlabel(r'$\rm E_{\rm FRET}(t)$',fontsize=14./self.canvas.devicePixelRatio())
			popplot.ax.set_ylabel('Probability',fontsize=14./self.canvas.devicePixelRatio())
			for asp in ['top','bottom','left','right']:
				popplot.ax.spines[asp].set_linewidth(1.0/self.canvas.devicePixelRatio())
			popplot.f.subplots_adjust(left=.13,bottom=.15,top=.95,right=.99)
			popplot.f.canvas.draw()

	def hist2d(self):
		if self.docks['plots_2D'][0].isHidden():
			self.docks['plots_2D'][0].show()
		self.docks['plots_2D'][0].raise_()
		popplot = self.docks['plots_2D'][1]
		popplot.clf()

		if self.ncolors == 2:
			fpb = self.get_plot_data()[0]
			if self.gui.prefs['synchronize_start_flag'] == 'True':
				print np.nansum(fpb)
				for i in range(fpb.shape[0]):
					y = fpb[i].copy()
					fpb[i] = np.nan
					pre = self.pre_list[i]
					post = self.pb_list[i]
					if pre < post:
						fpb[i,0:post-pre] = y[pre:post]
				print np.nansum(fpb)
			elif not self.hmm_result is None:
				state,success = QInputDialog.getInt(self,"Pick State","Which State?",min=0,max=self.hmm_result.nstates-1)
				if success:
					v = self.get_viterbi_data()
					vv = np.array([[v[i,:-1],v[i,1:]] for i in range(v.shape[0])])
					oo = []
					for i in range(fpb.shape[0]):
						ms = np.nonzero((vv[i,1]==state)*(vv[i,0]!=vv[i,1]))[0]
						if v[i,0] == state:
							ms = np.append(0,ms)
						ms = np.append(ms,v.shape[1])

						for j in range(ms.size-1):
							o = fpb[i].copy()
							ox = int(np.max((0,ms[j]-self.gui.prefs['plotter_2d_syncpreframes'])))
							o = o[ox:ms[j+1]]
							ooo = np.empty(v.shape[1]) + np.nan
							ooo[:o.size] = o
							oo.append(ooo)
					fpb = np.array(oo)


			dtmin = self.gui.prefs['plotter_min_time']
			dtmax = self.gui.prefs['plotter_max_time']
			if dtmax == -1:
				dtmax = fpb.shape[1]
			dt = np.arange(dtmin,dtmax)*self.gui.prefs['tau']
			ts = np.array([dt for _ in range(fpb.shape[0])])
			fpb = fpb[:,dtmin:dtmax]
			xcut = np.isfinite(fpb)
			bt = (self.gui.prefs['plotter_max_fret'] - self.gui.prefs['plotter_min_fret']) / (self.gui.prefs['plotter_nbins_fret'] + 1)
			z,hx,hy = np.histogram2d(ts[xcut],fpb[xcut],bins = [self.gui.prefs['plotter_nbins_time'],self.gui.prefs['plotter_nbins_fret']+2],range=[[dt[0],dt[-1]],[self.gui.prefs['plotter_min_fret']-bt,self.gui.prefs['plotter_max_fret']+bt]])
			rx = hx[:-1]
			ry = .5*(hy[1:]+hy[:-1])
			x,y = np.meshgrid(rx,ry,indexing='ij')

			from scipy.ndimage import gaussian_filter
			z = gaussian_filter(z,(self.gui.prefs['plotter_smoothx'],self.gui.prefs['plotter_smoothy']))

			# cm = plt.cm.rainbow
			# vmin = self.gui.prefs['plotter_floor']
			# cm.set_under('w')
			# if vmin <= 1e-300:
			# 	vmin = z.min()
			# pc = plt.pcolor(y.T,x.T,z.T,cmap=cm,vmin=vmin,edgecolors='face')
			try:
				cm = plt.cm.__dict__[self.gui.prefs['plotter_cmap']]
			except:
				cm = plt.cm.rainbow
			try:
				cm.set_under(self.gui.prefs['plotter_floorcolor'])
			except:
				cm.set_under('w')

			vmin = self.gui.prefs['plotter_floor']

			if self.gui.prefs['plotter_2d_normalizecolumn'] == 'True':
				z /= np.nanmax(z,axis=1)[:,None]
			else:
				z /= np.nanmax(z)

			z = np.nan_to_num(z)

			x -= self.gui.prefs['plotter_timeshift']

			from matplotlib.colors import LogNorm
			if vmin <= 0 or vmin >=z.max():
				pc = popplot.ax.contourf(x.T,y.T,z.T,self.gui.prefs['plotter_nbins_contour'],cmap=cm)
			else:
				# pc = plt.pcolor(y.T,x.T,z.T,vmin =vmin,cmap=cm,edgecolors='face',lw=1,norm=LogNorm(z.min(),z.max()))
				pc = popplot.ax.contourf(x.T,y.T,z.T,self.gui.prefs['plotter_nbins_contour'],vmin =vmin,cmap=cm)
			for pcc in pc.collections:
				pcc.set_edgecolor("face")

			try:
				cb = popplot.f.colorbar(pc)
				cb.set_ticks(np.array((0.,.2,.4,.6,.8,1.)))
				cb.ax.yaxis.set_tick_params(labelsize=12./self.canvas.devicePixelRatio(),direction='in',width=1.0/self.canvas.devicePixelRatio(),length=4./self.canvas.devicePixelRatio())
				for asp in ['top','bottom','left','right']:
					cb.ax.spines[asp].set_linewidth(1.0/self.canvas.devicePixelRatio())
				cb.solids.set_edgecolor('face')
				cb.solids.set_rasterized(True)
			except:
				pass

			for asp in ['top','bottom','left','right']:
				popplot.ax.spines[asp].set_linewidth(1.0/self.canvas.devicePixelRatio())
			popplot.f.subplots_adjust(left=.18,bottom=.14,top=.95,right=.99)

			popplot.ax.set_xlim(rx.min()-self.gui.prefs['plotter_timeshift'],rx.max()-self.gui.prefs['plotter_timeshift'])
			popplot.ax.set_ylim(self.gui.prefs['plotter_min_fret'],self.gui.prefs['plotter_max_fret'])
			popplot.ax.set_xlabel('Time (s)',fontsize=14./self.canvas.devicePixelRatio())
			popplot.ax.set_ylabel(r'$\rm E_{\rm FRET}(t)$',fontsize=14./self.canvas.devicePixelRatio())
			bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=1./self.canvas.devicePixelRatio())
			popplot.ax.annotate('n = %d'%(fpb.shape[0]),xy=(.95,.93),xycoords='axes fraction',ha='right',color='k',bbox=bbox_props,fontsize=12./self.canvas.devicePixelRatio())
			popplot.canvas.draw()

	def tdplot(self):
		if self.docks['plots_TDP'][0].isHidden():
			self.docks['plots_TDP'][0].show()
		self.docks['plots_TDP'][0].raise_()
		popplot = self.docks['plots_TDP'][1]
		popplot.clf()

		if self.ncolors == 2:
			fpb = self.get_plot_data()[0]
			d = np.array([[fpb[i,:-1],fpb[i,1:]] for i in range(fpb.shape[0])])

			if not self.hmm_result is None:
				# state,success = QInputDialog.getInt(self,"Pick State","Which State?",min=0,max=self.hmm_result.nstates-1)
				# if success:
				v = self.get_viterbi_data()
				vv = np.array([[v[i,:-1],v[i,1:]] for i in range(v.shape[0])])

				for i in range(d.shape[0]):
					d[i,:,vv[i,0]==vv[i,1]] = np.array((np.nan,np.nan))

			rx = np.linspace(self.gui.prefs['plotter_min_fret'],self.gui.prefs['plotter_max_fret'],self.gui.prefs['plotter_nbins_fret'])
			ry = np.linspace(self.gui.prefs['plotter_min_fret'],self.gui.prefs['plotter_max_fret'],self.gui.prefs['plotter_nbins_fret'])
			x,y = np.meshgrid(rx,ry,indexing='ij')
			dx = d[:,0].flatten()
			dy = d[:,1].flatten()
			cut = np.isfinite(dx)*np.isfinite(dy)
			z,hx,hy = np.histogram2d(dx[cut],dy[cut],bins=[rx.size,ry.size],range=[[rx.min(),rx.max()],[ry.min(),ry.max()]])

			from scipy.ndimage import gaussian_filter
			z = gaussian_filter(z,(self.gui.prefs['plotter_smoothx'],self.gui.prefs['plotter_smoothy']))

			try:
				cm = plt.cm.__dict__[self.gui.prefs['plotter_cmap']]
			except:
				cm = plt.cm.rainbow
			try:
				cm.set_under(self.gui.prefs['plotter_floorcolor'])
			except:
				cm.set_under('w')

			from matplotlib.colors import LogNorm
			if self.gui.prefs['plotter_floor'] <= 0:
				bins = np.logspace(np.log10(z[z>0.].min()),np.log10(z.max()),self.gui.prefs['plotter_nbins_contour'])
				pc = popplot.ax.contourf(x, y, z, bins, cmap=cm, norm=LogNorm())
			else:
				z[z< 1e-10] = 1e-9
				bins = np.logspace(0,np.log10(z.max()),self.gui.prefs['plotter_nbins_contour'])
				bins = np.append(1e-10,bins)
				pc = popplot.ax.contourf(x, y, z, bins, vmin=self.gui.prefs['plotter_floor'], cmap=cm, norm=LogNorm())

			for pcc in pc.collections:
				pcc.set_edgecolor("face")

			cb = popplot.f.colorbar(pc)
			zm = np.floor(np.log10(z.max()))
			cz = np.logspace(0,zm,zm+1)
			cb.set_ticks(cz)
			# cb.set_ticklabels(cz)
			cb.ax.yaxis.set_tick_params(labelsize=12./self.canvas.devicePixelRatio(),direction='in',width=1.0/self.canvas.devicePixelRatio(),length=4./self.canvas.devicePixelRatio())
			for asp in ['top','bottom','left','right']:
				cb.ax.spines[asp].set_linewidth(1.0/self.canvas.devicePixelRatio())
			cb.solids.set_edgecolor('face')
			cb.solids.set_rasterized(True)

			popplot.ax.set_xlim(rx.min(),rx.max())
			popplot.ax.set_ylim(ry.min(),ry.max())
			popplot.ax.set_xlabel(r'Initial E$_{\rm FRET}$',fontsize=14./self.canvas.devicePixelRatio())
			popplot.ax.set_ylabel(r'Final E$_{\rm FRET}$',fontsize=14./self.canvas.devicePixelRatio())
			popplot.ax.set_title('Transition Density (Counts)',fontsize=12/self.canvas.devicePixelRatio())
			bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=1./self.canvas.devicePixelRatio())
			popplot.ax.annotate('n = %d'%(fpb.shape[0]),xy=(.95,.93),xycoords='axes fraction',ha='right',color='k',bbox=bbox_props,fontsize=12./self.canvas.devicePixelRatio())

			for asp in ['top','bottom','left','right']:
				popplot.ax.spines[asp].set_linewidth(1.0/self.canvas.devicePixelRatio())
			popplot.f.subplots_adjust(left=.18,bottom=.14,top=.92,right=.99)

			popplot.f.canvas.draw()

	def update_colors(self):
		for i in range(self.ncolors):
			color = self.gui.prefs['channel_colors'][i]
			for j in range(3):
				self.a[0][0].lines[3*i+j].set_color(color)
			self.a[0][1].lines[i].set_color(color)
		for i in range(self.ncolors-1):
			if self.ncolors == 2:
				color = 'blue'
			else:
				color = self.gui.prefs['channel_colors'][i+1]
			for j in range(3):
				self.a[1][0].lines[3*i+j].set_color(color)
			self.a[1][1].lines[i].set_color(color)

	## Plot current trajectory
	def update(self):
		intensities = self.d[self.index].copy()

		if self.gui.prefs['convert_flag']:
			for i in range(self.ncolors):
				intensities[i] = self.gui.prefs['convert_c_lambda'][i]/self.gui.prefs['convert_em_gain']*(intensities[i] - self.gui.prefs['convert_offset'])

		bts = self.gui.prefs['bleedthrough'].reshape((4,4))
		for i in range(self.ncolors):
			for j in range(self.ncolors):
				intensities[j] -= bts[i,j]*intensities[i]


		t = np.arange(intensities.shape[1])*self.gui.prefs['tau']

		downsample = int(self.gui.prefs['downsample'])
		ll = t.size / downsample
		intensities = np.array([np.sum(intensities[i,:ll*downsample].reshape((ll,downsample)),axis=1) for i in range(self.ncolors)])
		t = t[:ll*downsample].reshape((ll,downsample))[:,0]
		pbtime = int(self.pb_list[self.index] / downsample)
		pretime = int(self.pre_list[self.index] / downsample)

		for i in range(self.ncolors):
			self.a[0][0].lines[3*i+0].set_data(t[:pretime],intensities[i,:pretime])
			self.a[0][0].lines[3*i+1].set_data(t[pretime:pbtime],intensities[i,pretime:pbtime])
			self.a[0][0].lines[3*i+2].set_data(t[pbtime:],intensities[i,pbtime:])

		for i in range(self.ncolors-1):
			rel = intensities[1:] / (1e-300+intensities.sum(0)[None,:])
			self.a[1][0].lines[3*i+0].set_data(t[:pretime],rel[i,:pretime])
			self.a[1][0].lines[3*i+1].set_data(t[pretime:pbtime],rel[i,pretime:pbtime])
			self.a[1][0].lines[3*i+2].set_data(t[pbtime:],rel[i,pbtime:])

		if not self.hmm_result is None:
			if self.hmm_result.ran.count(self.index)>0:
				ii = self.hmm_result.ran.index(self.index)
				if self.gui.prefs['hmm_binding_expt'] == 'True':
					mmmm = np.array((0.,1.))
					vvvv = self.hmm_result.viterbi[ii]

					self.a[1,0].lines[-1].set_data(t[pretime:pbtime],mmmm[vvvv])
				else:
					self.a[1,0].lines[-1].set_data(t[pretime:pbtime],self.hmm_result.m[self.hmm_result.viterbi[ii]])
			else:
				self.a[1,0].lines[-1].set_data([0,0],[0,0])


		hymaxes = []
		for i in range(self.ncolors):
			if pretime < pbtime:
				hy,hx = np.histogram(intensities[i,pretime:pbtime],range=self.yminmax,bins=int(np.sqrt(pbtime-pretime)))
			else:
				hy = np.zeros(100)
				hx = np.linspace(self.yminmax[0],self.yminmax[1],101)
			hy = np.append(np.append(0.,hy),0.)
			hx = np.append(np.append(hx[0],.5*(hx[1:]+hx[:-1])),hx[-1])
			self.a[0][1].lines[i].set_data(hy,hx)
			hymaxes.append(hy.max())

		for i in range(self.ncolors-1):
			if pretime < pbtime:
				hy,hx = np.histogram(rel[i,pretime:pbtime],range=(self.gui.prefs['plotter_min_fret'],self.gui.prefs['plotter_max_fret']),bins=int(np.sqrt(pbtime-pretime)))
			else:
				hy = np.zeros(100)
				hx = np.linspace(self.yminmax[0],self.yminmax[1],101)
			hy = np.append(np.append(0.,hy),0.)
			hx = np.append(np.append(hx[0],.5*(hx[1:]+hx[:-1])),hx[-1])
			self.a[1][1].lines[i].set_data(hy,hx)
			hymaxes.append(hy.max())

		self.update_colors()

		self.a[0][0].set_xlim(0,t[-1])
		self.a[1][0].set_ylim(self.gui.prefs['plotter_min_fret'],self.gui.prefs['plotter_max_fret'])
		self.a[0][1].set_xlim(1, np.max(hymaxes)*1.25)
		self.a[1][1].set_xlim(self.a[0][1].get_xlim())

		# mmin = np.nanmin((donor.min,acceptor.min()))
		# mmax = np.nanmax((donor.max(),acceptor.max()))
		mmin,mmax = self.yminmax
		self.a[0][0].set_ylim(mmin,mmax)
		# delta = mmax-mmin
		# self.a[0][0].set_ylim( mmin - delta*.25, mmax + delta*.25)

		self.a[0][0].set_title(str(self.index)+' / '+str(self.d.shape[0] - 1) + " -  %d"%(self.class_list[self.index]),fontsize=12./self.canvas.devicePixelRatio())

		self.a[0][0].yaxis.set_label_coords(-.18, 0.5)
		self.a[1][0].yaxis.set_label_coords(-.18, 0.5)
		self.a[1][0].xaxis.set_label_coords(0.5, -.21)
		self.a[1][1].xaxis.set_label_coords(0.5, -.21)
		self.canvas.draw()

	## Calculate the anti-correlation for sorting traces
	def cross_corr_order(self):
		x = self.d[:,0] #- self.d[:,0].mean(1)[:,None]
		y = self.d[:,1] #- self.d[:,1].mean(1)[:,None]
		x = np.gradient(x,axis=1)
		y = np.gradient(y,axis=1)

		a = np.fft.fft(x,axis=1)
		b = np.conjugate(np.fft.fft(y,axis=1))
		order = np.fft.ifft((a*b),axis=1)
		order = order[:,0].real.argsort()
		return order

	def batch_load(self):
		try:
			if not self.ui_batch.isVisible():
				self.ui_batch.setVisible(True)
			self.ui_batch.raise_()
		except:
			self.ui_batch = ui_batch_loader(self)
			self.ui_batch.setWindowTitle('Batch Load')
			self.ui_batch.show()


class deleting_list(QListWidget):
	def __init__(self):
		super(deleting_list,self).__init__()
		self.setSelectionMode(QAbstractItemView.ExtendedSelection)
		self.setDragEnabled(True)
		self.setDropIndicatorShown(True)
		self.setDragDropMode(QAbstractItemView.InternalMove)

	def keyPressEvent(self,event):
		if event.key() == Qt.Key_Backspace or event.key() == Qt.Key_Delete:
			sis = self.selectedItems()
			if len(sis) > 0:
				for si in sis:
					self.takeItem(self.row(si))
		QListWidget.keyPressEvent(self,event)
		event.accept()

	def load_files(self):
		fname = QFileDialog.getOpenFileNames(self,'Choose Files','./')
		if len(fname[0]) > 0:
			self.addItems(fname[0])

class ui_batch_loader(QMainWindow):
	def __init__(self,parent=None):
		super(QMainWindow,self).__init__(parent)
		self.ui = batch_loader(self)
		self.setCentralWidget(self.ui)
		self.show()

	def closeEvent(self,event):
		self.parent().activateWindow()
		self.parent().raise_()
		self.parent().setFocus()

class batch_loader(QWidget):
	def __init__(self,parent=None):
		super(QWidget,self).__init__(parent=parent)
		self.initialize()
		self.window = self.parent().parent()

	def initialize(self):
		tothbox = QHBoxLayout()

		widl1 = QWidget()
		vbox = QVBoxLayout()
		self.l1 = deleting_list()
		vbox.addWidget(QLabel('Trajectories'))
		vbox.addWidget(self.l1)
		hw = QWidget()
		hbox = QHBoxLayout()
		b1a = QPushButton('Add')
		b1c = QPushButton('Clear')
		hbox.addStretch()
		hbox.addWidget(b1a)
		hbox.addWidget(b1c)
		hw.setLayout(hbox)
		vbox.addWidget(hw)
		b1a.clicked.connect(self.l1.load_files)
		b1c.clicked.connect(self.l1.clear)
		widl1.setLayout(vbox)

		widl2 = QWidget()
		vbox = QVBoxLayout()
		self.l2 = deleting_list()
		vbox.addWidget(QLabel('Classes/Bleaching'))
		vbox.addWidget(self.l2)
		hw = QWidget()
		hbox = QHBoxLayout()
		b2a = QPushButton('Add')
		b2c = QPushButton('Clear')
		hbox.addStretch()
		hbox.addWidget(b2a)
		hbox.addWidget(b2c)
		hw.setLayout(hbox)
		vbox.addWidget(hw)
		b2a.clicked.connect(self.l2.load_files)
		b2c.clicked.connect(self.l2.clear)
		widl2.setLayout(vbox)

		vwid = QWidget()
		vboxsub = QVBoxLayout()
		ss = 'Bulk Import\nPick trajectories and matching class files\nfor bulk import.  Match corresponding files\nin the same row by dragging and dropping.\nRemove files with delete key or clear. If\nyou do not have class files, keep the class\npane empty. Click the load button to load.'
		bsub = QPushButton('Load')
		vboxsub.addWidget(QLabel(ss))
		vboxsub.addWidget(bsub)
		vwid.setLayout(vboxsub)
		bsub.clicked.connect(self.batch_load)

		tothbox.addWidget(widl1)
		tothbox.addWidget(widl2)
		tothbox.addWidget(vwid)

		self.setLayout(tothbox)

	def batch_load(self):
		count1 = self.l1.count()
		count2 = self.l2.count()
		if count1 == 0:
			return None
		elif (count2 == 0) or (count1 == count2):
			ltraj = []
			lclass = []
			for i in range(count1):
				ltraj.append(self.l1.item(i).text())
				if count2 != 0:
					lclass.append(self.l2.item(i).text())
				else:
					lclass.append(None)

			ncolors,suc2 = QInputDialog.getInt(self,"Number of Color Channels","Number of Color Channels",value=2,min=1)
			if suc2:
				ds = []
				cs = []
				for i in range(len(ltraj)):
					try:
						d = np.loadtxt(ltraj[i],delimiter=',').T
						dd = np.array([d[j::ncolors] for j in range(ncolors)])
						d = np.moveaxis(dd,1,0)

						if not lclass[i] is None:
							c = np.loadtxt(lclass[i],delimiter=',').astype('i')
						else:
							c = np.zeros((d.shape[0],1+2*ncolors),dtype='i')
							c[:,2::2] = d.shape[2]
						ds.append(d)
						cs.append(c)
					except:
						print "Could not load:\n\t%s\n\t%s"%(ltraj[i],lclass[i])

				if len(cs) > 0:
					cc = np.concatenate(cs,axis=0)
					maxlength = np.max([dd.shape[2] for dd in ds]).astype('i')
					for i in range(len(ds)):
						dsi = ds[i]
						dtemp = np.zeros((dsi.shape[0],dsi.shape[1],maxlength))
						dtemp[:,:,:dsi.shape[2]] = dsi
						ds[i] = dtemp
					dd = np.concatenate(ds,axis=0)

					self.window.ncolors = ncolors
					self.window.initialize_data(dd,sort=False)
					self.window.index = 0
					self.window.initialize_plots()
					self.window.initialize_sliders()

					self.window.class_list = cc[:,0]
					self.window.pre_list = cc[:,1::2].max(1)
					self.window.pb_list = cc[:,2::2].min(1)
					self.window.update()

					self.window.ui_batch.close()
					QMessageBox.information(self,"Batch Load","Loaded %d pairs of files containing a total of %d trajectories."%(len(ds),dd.shape[0]))
				else:
					QMessageBox.critical(self,"Batch Load","Could not find any traces in these files")


class fake(object):
	pass

class mpl_plot(QWidget):
	def __init__(self):
		super(QWidget,self).__init__()

		self.f,self.ax = plt.subplots(1,figsize=(6,4))
		self.canvas = FigureCanvas(self.f)
		self.toolbar = NavigationToolbar(self.canvas,None)

		sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
		self.canvas.setSizePolicy(sizePolicy)
		self.f.set_dpi(self.f.get_dpi()/self.canvas.devicePixelRatio())
		self.fix_ax()

		self.canvas.draw()
		plt.close(self.f)

		layout = QVBoxLayout()
		layout.addWidget(self.canvas)
		layout.addWidget(self.toolbar)
		layout.addStretch()
		self.setLayout(layout)

	def fix_ax(self):
		offset = .08
		offset2 = 0.14
		self.f.subplots_adjust(left=offset2,right=1.-offset,top=1.-offset,bottom=offset2)
		self.ax.tick_params(labelsize=12./self.canvas.devicePixelRatio(),axis='both',direction='in',width=1.0/self.canvas.devicePixelRatio(),length=4./self.canvas.devicePixelRatio())

		self.ax.tick_params(axis='both', which='major', labelsize=12./self.canvas.devicePixelRatio())
		self.ax.format_coord = lambda x, y: ''
		self.f.tight_layout()

	def clf(self):
		self.f.clf()
		self.ax = self.f.add_subplot(111)
		self.fix_ax()

def launch():
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
	import sys
	app = QApplication([])
	app.setStyle('fusion')

	g = ui_plotter(None,[])
	app.setWindowIcon(g.windowIcon())
	if __name__ == '__main__':
		sys.exit(app.exec_())
	else:
		return g

## Run from the command line with `python plotter.py`
if __name__ == '__main__':
	launch()
