from PyQt5.QtWidgets import QMainWindow, QWidget, QSizePolicy, QVBoxLayout, QShortcut, QSlider, QHBoxLayout, QPushButton, QFileDialog, QCheckBox,QApplication, QAction,QLineEdit,QLabel,QGridLayout, QInputDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

import numpy as np
from supporting.photobleaching import get_point_pbtime, pb_ensemble, pb_snr, remove_pb_all
from supporting import simul_vbem_hmm as hmm


class fake(object):
	pass


## GUI for plotting 2D smFRET trajectories
class ui_plotter(QMainWindow):
	def __init__(self,data,spots,bg=None,parent=None,):
		super(QMainWindow,self).__init__(parent)

		## If there's no vbscope movie GUI, then still need to have access to the preferences
		if not parent is None:
			self.gui = parent.gui
		else:
			self.gui = fake()
			self.gui.prefs = {'bleedthrough':.04,'tau':.1,'downsample':1,'pb_length':10,'snr_threshold':1.}

		## Make the plotter widget
		self.ui = plotter(data,self)
		self.setCentralWidget(self.ui)
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

		self.gui = parent.gui
		layout = QVBoxLayout()

		## Initialize Plots
		self.f,self.a = plt.subplots(2,2,gridspec_kw={'width_ratios':[6,1]},figsize=(6.5,4))
		self.canvas = FigureCanvas(self.f)
		sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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
		twidg = QWidget()
		hbox = QHBoxLayout()
		hbox.addWidget(self.slider_select)
		twidg.setLayout(hbox)
		layout.addWidget(self.canvas)
		layout.addWidget(self.toolbar)
		layout.addWidget(twidg)
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

		lw=.75
		pb=.2

		## plot pre-truncated, kept, and post-truncated trajectory (Intensities)
		self.a[0][0].plot(np.random.rand(self.d.shape[0]),color='g',ls=':',alpha=pb,lw=lw)
		self.a[0][0].plot(np.random.rand(self.d.shape[0]),color='g',alpha=.8,lw=lw)
		self.a[0][0].plot(np.random.rand(self.d.shape[0]),color='g',ls=':',alpha=pb,lw=lw)
		self.a[0][0].plot(np.random.rand(self.d.shape[0]),color='r',ls=':',alpha=pb,lw=lw)
		self.a[0][0].plot(np.random.rand(self.d.shape[0]),color='r',alpha=.8,lw=lw)
		self.a[0][0].plot(np.random.rand(self.d.shape[0]),color='r',ls=':',alpha=pb,lw=lw)

		## plot pre-truncated, kept, and post-truncated trajectory (E_{FRET})
		self.a[1][0].plot(np.random.rand(self.d.shape[0]),color='b',ls=':',alpha=pb,lw=lw)
		self.a[1][0].plot(np.random.rand(self.d.shape[0]),color='b',alpha=.8,lw=lw)
		self.a[1][0].plot(np.random.rand(self.d.shape[0]),color='b',ls=':',alpha=pb,lw=lw)

		## Plot histograms of donor, acceptor, E_{FRET}
		self.a[0][1].plot(np.random.rand(100),color='g',alpha=.8)
		self.a[0][1].plot(np.random.rand(100),color='r',alpha=.8)
		self.a[1][1].plot(np.random.rand(100),color='b',alpha=.8)

		## Make it so that certain plots zoom together
		self.a[0][0].get_shared_y_axes().join(self.a[0][0],self.a[0][1])
		self.a[1][0].get_shared_y_axes().join(self.a[1][0],self.a[1][1])
		self.a[0][0].get_shared_x_axes().join(self.a[0][0],self.a[1][0])
		self.a[0][1].get_shared_x_axes().join(self.a[0][1],self.a[1][1])

		## Set the ticks/labels so that they look nice
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
		self.f.subplots_adjust(hspace=.03,wspace=0.015)
		self.f.canvas.draw()

	## Reset everything with the new trajectories loaded in with this function
	def initialize_data(self,data):
		self.d = data

		## Guess at good y-limits for the plot
		self.yminmax = np.percentile(self.d.flatten(),[.1,99.9])
		self.le_min.setText(str(self.yminmax[0]))
		self.le_max.setText(str(self.yminmax[1]))

		## Sort trajectories based on anti-correlation (most is first, least is last)
		order = self.cross_corr_order()
		self.d = self.d[order]

		## Calculate/set photobleaching, initialize class list
		q = np.copy(self.d)
		q[:,1] -= self.gui.prefs['bleedthrough']*q[:,0]
		self.fret = q[:,1]/(q[:,0] + q[:,1])
		# self.fret = pb.remove_pb_all(self.fret)
		# cut = np.isnan(self.fret)
		# self.d[:,0][cut] = np.nan
		# self.d[:,0][cut] = np.nan
		print self.d.shape
		self.pre_list = np.zeros(self.d.shape[0],dtype='i')
		self.pb_list = self.pre_list.copy() + self.d.shape[2]
		self.class_list = np.zeros(self.d.shape[0])

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

		file_exit = QAction('Exit', self, shortcut='Ctrl+Q')
		file_exit.triggered.connect(self.parent().close)

		for f in [file_load_traces,file_load_classes,file_exit]:
			menu_file.addAction(f)

		### save
		menu_save = self.menubar.addMenu('Export')

		export_traces = QAction('Save Traces', self, shortcut='Ctrl+S')
		export_traces.triggered.connect(self.export_traces)

		export_classes = QAction('Save Classes', self, shortcut='Ctrl+D')
		export_classes.triggered.connect(self.export_classes)

		for f in [export_traces,export_classes]:
			menu_save.addAction(f)

		### tools
		menu_tools = self.menubar.addMenu('Tools')

		tools_cull = QAction('Cull SNR', self)
		tools_cull.triggered.connect(self.cull_snr)
		tools_cullpb = QAction('Cull PB', self)
		tools_cullpb.triggered.connect(self.cull_pb)
		tools_step = QAction('Photobleach - Step',self)
		tools_step.triggered.connect(self.photobleach_step)
		tools_var = QAction('Remove all PB - Var',self)
		tools_var.setCheckable(True)
		tools_var.setChecked(False)
		self.pb_remove_check = tools_var
		tools_hmm = QAction('HMM',self)
		tools_hmm.triggered.connect(self.run_hmm)

		for f in [tools_cull,tools_cullpb,tools_step,tools_var,tools_hmm]:
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

	def photobleach_step(self):
		if not self.d is None:
			q = np.copy(self.d)
			q[:,1] -= self.gui.prefs['bleedthrough']*q[:,0]
			# l1 = calc_pb_time(self.fret,self.gui.prefs['pb_length'])
			l2 = pb_ensemble(q[:,0] + q[:,1])[1]
			# self.pb_list = np.array([np.min((l1[i],l2[i])) for i in range(l1.size)])
			self.pb_list = l2
			self.update()

	def run_hmm(self):
		if not self.d is None:
			nstates,success = QInputDialog.getInt(self,"Number of HMM States","Number of HMM States")
			if success and nstates > 1:
				y = []
				checked = self.get_checked()
				ran = []
				for i in range(self.fret.shape[0]):
					if checked[i]:
						yy = self.fret[i,self.pre_list[i]:self.pb_list[i]]
						yy[np.isnan(yy)] = -1.
						yy[yy < -1.] = -1.
						yy[yy > 2] = 2.
						if yy.size > 5:
							y.append(yy)
							ran.append(i)
				nrestarts = 8
				priors = [hmm.initialize_priors(y,nstates) for _ in range(nrestarts)]

				result,lbs = hmm.hmm(y,nstates,priors,nrestarts)
				print lbs
				print '\nHMM - k = %d, iter= %d, lowerbound=%f'%(nstates,result.iterations,result.lowerbound)
				print '  m:',result.m
				print 'sig:',(result.b/result.a)**.5
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

	## Remove trajectories with SNR less than threshold
	def cull_snr(self):
		snr_threshold = self.gui.prefs['snr_threshold']

		# snr = np.zeros(self.d.shape[0])
		# for i in range(snr.size):
		# 	q = self.d[i].sum(0)
		# 	snr[i] = q.mean()/np.std(q)
		snr = pb_snr(self.d.sum(1))
		cut = snr > snr_threshold
		print "kept %d out of %d = %f"%(cut.sum(),snr.size,cut.sum()/float(snr.size))

		plt.figure()
		x = np.linspace(0.,15,131)
		y = np.zeros_like(x)
		for i in range(y.size):
			y[i] = (snr > x[i]).sum()
		y/=y[0]
		plt.plot(x,y)
		plt.axvline(snr_threshold,color='r')
		plt.show()

		d = self.d[cut]
		self.index = 0
		self.initialize_data(d)
		self.initialize_plots()
		self.initialize_sliders()

	## Remove trajectories with number of kept-frames < threshold
	def cull_pb(self):
		pbt = self.pb_list
		pret = self.pre_list
		dt = pbt-pret
		cut = dt > self.gui.prefs['pb_length']
		print "kept %d out of %d = %f"%(cut.sum(),pbt.size,cut.sum()/float(pbt.size))
		d = self.d[cut]
		self.index = 0
		self.initialize_data(d)
		self.initialize_plots()
		self.initialize_sliders()

	## Save raw donor-acceptor trajectories (bleedthrough corrected) in vbscope format (commas)
	def export_traces(self):
		n = 2 #number of colors
		if not self.d is None:
			dd = self.d.copy()
			if n == 2:
				dd[:,1] -= self.gui.prefs['bleedthrough']*dd[:,0]

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
		n = 2 #number of colors
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
			try:
				d = np.loadtxt(fname[0],delimiter=',').T
				dd = np.array([d[::2],d[1::2]])
				d = np.moveaxis(dd,1,0)

				success = True
			except:
				print "could not load %s"%(fname[0])

			if success:
				self.initialize_data(d)
				self.initialize_plots()
				self.initialize_sliders()

	## Try to load classes/bleach times from a vbscope style file (commas) (Nx5)
	def load_classes(self):
		fname = QFileDialog.getOpenFileName(self,'Choose file to load classes','./')#,filter='TIF File (*.tif *.TIF)')
		if fname[0] != "":
			success = False
			try:
				d = np.loadtxt(fname[0],delimiter=',').astype('i')
				if d.shape[1] == 5 and d.shape[0] == self.d.shape[0]:
					success = True
			except:
				print "could not load %s"%(fname[0])

			if success:
				self.class_list = d[:,0]
				self.pre_list = np.array([d[:,1],d[:,3]]).max(0)
				self.pb_list = np.array([d[:,2],d[:,4]]).min(0)
				self.update()

	## Handler for mouse clicks in main plots
	def mouse_click(self,event):
		if (event.inaxes == self.a[0][0] or event.inaxes == self.a[1][0]) and not self.d is None:
			## Right click - set photobleaching point
			if event.button == 3 and self.toolbar._active is None:
				self.pb_list[self.index] = int(np.round(event.xdata/self.gui.prefs['tau']))
				self.update()
			## Left click - set pre-truncation point
			if event.button == 1 and self.toolbar._active is None:
				self.pre_list[self.index] = int(np.round(event.xdata/self.gui.prefs['tau']))
				self.update()
			## Middle click - reset pre and post points to calculated values
			if event.button == 2 and self.toolbar._active is None:
				self.pre_list[self.index] = 0
				self.pb_list[self.index] = get_point_pbtime(self.d[self.index].sum(0))
				self.update()

	## Add axis labels to plots
	def label_axes(self):
		fs = 12
		self.a[0][0].set_ylabel(r'Intensity (a.u.)',fontsize=fs,va='top')
		self.a[1][0].set_ylabel(r'E$_{\rm{FRET}}$',fontsize=fs,va='top')
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

		if kk == 'h':
			self.ensemble_hist()

		if np.any(np.arange(10)==kk):
			self.class_list[self.index] = kk
			self.update()

		try:
			self.gui.app.processEvents()
		except:
			pass
		# self.update()

	def get_plot_data(self):
		# f = self.fret
		fpb = self.fret.copy()

		for i in range(fpb.shape[0]):
			fpb[i,:self.pre_list[i]] = np.nan
			fpb[i,self.pb_list[i]:] = np.nan
		checked = self.get_checked()
		fpb = fpb[checked]
		if self.pb_remove_check.isChecked():
			fpb = remove_pb_all(fpb)

		return fpb

	def get_checked(self):
		checked = np.zeros(self.d.shape[0],dtype='bool')
		for i in range(10):
			if self.action_classes[i].isChecked():
				checked[np.nonzero(self.class_list == i)] = True
		return checked

	## Plot the 1D Histogram of the ensemble
	def hist1d(self):
		fpb = self.get_plot_data()

		plt.figure(5)
		# plt.hist(f.flatten(),bins=181,range=(-.4,1.4),histtype='stepfilled',alpha=.8,normed=True)
		plt.hist(fpb.flatten(),bins=181,range=(-.4,1.4),histtype='stepfilled',alpha=.8,normed=True)
		plt.xlim(-.4,1.4)
		plt.xlabel(r'$\rm E_{\rm FRET}(t)$',fontsize=14)
		plt.ylabel('Probability',fontsize=14)
		plt.tight_layout()
		plt.show()

	def hist2d(self):
		fpb = self.get_plot_data()

		plt.figure(6)
		rx = np.linspace(-.4,1.4,81)
		ry = np.arange(fpb.shape[1])*self.gui.prefs['tau']
		x,y = np.meshgrid(rx,ry,indexing='ij')
		z = np.zeros_like(x)
		for i in range(ry.shape[0]):
			hy,hx = np.histogram(fpb[:,i],bins=rx.size,range=(rx.min(),rx.max()))
			z[:,i] = hy

		from scipy.ndimage import gaussian_filter
		z = gaussian_filter(z,(1,5))
		z/=z.max()
		cm = plt.cm.rainbow
		cm.set_under('w')
		vmin = 0
		pc = plt.pcolor(y.T,x.T,z.T,cmap=cm,vmin=vmin,edgecolors='face')
		cb = plt.colorbar()
		cb.solids.set_edgecolor('face')

		plt.xlim(0,ry.max())
		plt.xlabel('Time (s)',fontsize=14)
		plt.ylabel(r'$\rm E_{\rm FRET}(t)$',fontsize=14)
		bbox_props = dict(boxstyle="square", fc="w", alpha=1.0)
		plt.annotate('n = %d'%(fpb.shape[0]),xy=(.95,.95),xycoords='axes fraction',ha='right',color='k',bbox=bbox_props)
		plt.tight_layout()
		plt.show()

	def tdplot(self):
		fpb = self.get_plot_data()
		d = np.array([[fpb[i,:-1],fpb[i,1:]] for i in range(fpb.shape[0])])

		plt.figure(7)
		rx = np.linspace(-.4,1.4,81)
		ry = np.linspace(-.4,1.4,81)
		x,y = np.meshgrid(rx,ry,indexing='ij')
		dx = d[:,0].flatten()
		dy = d[:,1].flatten()
		cut = np.isfinite(dx)*np.isfinite(dy)
		z,hx,hy = np.histogram2d(dx[cut],dy[cut],bins=[rx.size,ry.size],range=[[rx.min(),rx.max()],[ry.min(),ry.max()]])

		from scipy.ndimage import gaussian_filter
		z = gaussian_filter(z,(.1,.1))
		cm = plt.cm.rainbow
		cm.set_under('w')
		vmin = 1
		from matplotlib.colors import LogNorm
		pc = plt.pcolor(y.T,x.T,z.T,vmin =vmin,cmap=cm,edgecolors='face',lw=1,norm=LogNorm(z.min(),z.max()))
		cb = plt.colorbar()
		cb.solids.set_edgecolor('face')
		cb.solids.set_rasterized(True)

		plt.xlim(rx.min(),rx.max())
		plt.ylim(ry.min(),ry.max())
		plt.xlabel(r'Initial E$_{\rm FRET}$',fontsize=14)
		plt.ylabel(r'Final E$_{\rm FRET}$',fontsize=14)
		plt.title('Transition Density (Counts)')
		bbox_props = dict(boxstyle="square", fc="w", alpha=1.0)
		plt.annotate('n = %d'%(fpb.shape[0]),xy=(.95,.95),xycoords='axes fraction',ha='right',color='k',bbox=bbox_props)
		plt.tight_layout()
		plt.show()

	## Plot current trajectory
	def update(self):
		pbp = self.gui.prefs['bleedthrough']
		downsample = 1#int(self.gui.prefs['downsample'])
		donor = self.d[self.index,0]
		acceptor = self.d[self.index,1] - pbp*self.d[self.index,0]
		t = np.arange(donor.size)*self.gui.prefs['tau']
		pbtime = self.pb_list[self.index]
		pretime = self.pre_list[self.index]

		ll = t.size / downsample
		donor = np.sum(donor[:ll*downsample].reshape((ll,downsample)),axis=1)
		acceptor = np.sum(acceptor[:ll*downsample].reshape((ll,downsample)),axis=1)
		t = t[:ll*downsample].reshape((ll,downsample))[:,0]

		# self.a[0].lines[0].set_data(t,donor)
		# self.a[0].lines[1].set_data(t,acceptor)

		self.a[0][0].lines[0].set_data(t[:pretime],donor[:pretime])
		self.a[0][0].lines[1].set_data(t[pretime:pbtime],donor[pretime:pbtime])
		self.a[0][0].lines[2].set_data(t[pbtime:],donor[pbtime:])
		self.a[0][0].lines[3].set_data(t[:pretime],acceptor[:pretime])
		self.a[0][0].lines[4].set_data(t[pretime:pbtime],acceptor[pretime:pbtime])
		self.a[0][0].lines[5].set_data(t[pbtime:],acceptor[pbtime:])
		# if not self.b is None:
		# 	bdonor = self.b[self.index,0]
		# 	bacceptor = self.b[self.index,1]
		# 	self.a[0].lines[0].set_data(np.arange(donor.size),bdonor)
		# 	self.a[0].lines[1].set_data(np.arange(donor.size),bacceptor)

		# self.a[1].lines[0].set_data(t,(acceptor/(donor+acceptor+1e-300)))
		self.a[1][0].lines[0].set_data(t[:pretime],(acceptor/(donor+acceptor+1e-300))[:pretime])
		self.a[1][0].lines[1].set_data(t[pretime:pbtime],(acceptor/(donor+acceptor+1e-300))[pretime:pbtime])
		self.a[1][0].lines[2].set_data(t[pbtime:],(acceptor/(donor+acceptor+1e-300))[pbtime:])

		if not self.hmm_result is None:
			if self.hmm_result.ran.count(self.index)>0:
				ii = self.hmm_result.ran.index(self.index)
				self.a[1,0].lines[3].set_data(t[pretime:pbtime],self.hmm_result.m[self.hmm_result.viterbi[ii]])
			else:
				self.a[1,0].lines[3].set_data([0,0],[0,0])


		# self.a[0][1].cla()
		# self.a[1][1].cla()
		if pretime < pbtime:
			hy1,hx = np.histogram(donor[pretime:pbtime],range=self.yminmax,bins=int(np.sqrt(pbtime-pretime)))
		else:
			hy1 = np.zeros(100)
			hx = np.linspace(self.yminmax[0],self.yminmax[1],101)
		hy1 = np.append(np.append(0.,hy1),0.)
		hx = np.append(np.append(hx[0],.5*(hx[1:]+hx[:-1])),hx[-1])
		self.a[0][1].lines[0].set_data(hy1,hx)

		if pretime < pbtime:
			hy2,hx = np.histogram(acceptor[pretime:pbtime],range=self.yminmax,bins=int(np.sqrt(pbtime-pretime)))
		else:
			hy2 = np.zeros(100)
			hx = np.linspace(self.yminmax[0],self.yminmax[1],101)
		hy2 = np.append(np.append(0.,hy2),0.)
		hx = np.append(np.append(hx[0],.5*(hx[1:]+hx[:-1])),hx[-1])
		self.a[0][1].lines[1].set_data(hy2,hx)

		if pretime < pbtime:
			hy3,hx = np.histogram((acceptor/(donor+acceptor+1e-300))[pretime:pbtime],range=(-.4,1.4),bins=int(np.sqrt(pbtime-pretime)))
		else:
			hy3 = np.zeros(100)
			hx = np.linspace(self.yminmax[0],self.yminmax[1],101)
		hy3 = np.append(np.append(0.,hy3),0.)
		hx = np.append(np.append(hx[0],.5*(hx[1:]+hx[:-1])),hx[-1])
		self.a[1][1].lines[0].set_data(hy3,hx)

		self.a[0][0].set_xlim(0,t[-1])
		self.a[1][0].set_ylim(-.4,1.4)
		self.a[0][1].set_xlim(1, np.max((hy1.max(),hy2.max(),hy3.max()))*1.25)
		self.a[1][1].set_xlim(self.a[0][1].get_xlim())

		# mmin = np.nanmin((donor.min,acceptor.min()))
		# mmax = np.nanmax((donor.max(),acceptor.max()))
		mmin,mmax = self.yminmax
		self.a[0][0].set_ylim(mmin,mmax)
		# delta = mmax-mmin
		# self.a[0][0].set_ylim( mmin - delta*.25, mmax + delta*.25)

		self.a[0][0].set_title(str(self.index)+' / '+str(self.d.shape[0] - 1) + " -  %d"%(self.class_list[self.index]))

		self.a[0][0].yaxis.set_label_coords(-.17, 0.5)
		self.a[1][0].yaxis.set_label_coords(-.17, 0.5)
		self.a[1][0].xaxis.set_label_coords(0.5, -.2)
		self.a[1][1].xaxis.set_label_coords(0.5, -.2)
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

## Launch the main window as a standalone GUI (ie without vbscope analyze movies)
def launch():
	import sys
	app = QApplication([])
	app.setStyle('fusion')

	g = ui_plotter(None,[])
	app.setWindowIcon(g.windowIcon())
	sys.exit(app.exec_())

## Run from the command line with `python plotter.py`
if __name__ == '__main__':
	launch()
