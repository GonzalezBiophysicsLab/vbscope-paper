from PyQt5.QtWidgets import QMainWindow, QWidget, QSizePolicy, QVBoxLayout, QShortcut, QSlider, QHBoxLayout, QPushButton, QFileDialog, QCheckBox,QApplication, QAction,QLineEdit,QLabel,QGridLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator

import numpy as np
from supporting.photobleaching import get_point_pbtime, calc_pb_time, pb_ensemble, pb_snr

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

class fake(object):
	pass

class ui_plotter(QMainWindow):
	def __init__(self,data,spots,bg=None,parent=None,):
		super(QMainWindow,self).__init__(parent)
		if not parent is None:
			self.gui = parent.gui
		else:
			self.gui = fake()
			self.gui.prefs = {'bleedthrough':.03,'tau':.1,'downsample':1,'pb_length':10,'snr_threshold':1.}
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

class plotter(QWidget):
	def __init__(self,data,parent=None):
		super(QWidget,self).__init__(parent=parent)

		self.gui = parent.gui
		layout = QVBoxLayout()

		self.f,self.a = plt.subplots(2,2,gridspec_kw={'width_ratios':[6,1]},figsize=(6.5,4))
		self.canvas = FigureCanvas(self.f)
		self.toolbar = NavigationToolbar(self.canvas,None)

		self.slider_select = QSlider(Qt.Horizontal)
		sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
		self.slider_select.setSizePolicy(sizePolicy)

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

		sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
		self.canvas.setSizePolicy(sizePolicy)

		twidg = QWidget()
		hbox = QHBoxLayout()
		hbox.addWidget(self.slider_select)
		# hbox.addWidget(self.checkbox_spot)
		# hbox.addWidget(self.button_export)
		twidg.setLayout(hbox)

		layout.addWidget(self.canvas)
		layout.addWidget(self.toolbar)
		layout.addWidget(twidg)
		layout.addWidget(scalewidget)
		self.setLayout(layout)

		self.setup_menubar()

		self.index = 0
		self.d = data
		if not self.d is None:
			self.initialize_data(data.astype('double'))
			self.initialize_plots()
			self.update()

		self.init_shortcuts()
		self.update_sliders()
		self.slider_select.valueChanged.connect(self.slide_switch)
		self.f.canvas.mpl_connect('button_press_event', self.mouse_click)
		plt.close(self.f)

	def update_minmax(self):
		self.yminmax = np.array((float(self.le_min.text()),float(self.le_max.text())))
		self.update()

	def update_sliders(self):
		self.slider_select.setMinimum(0)
		mm = 0
		if not self.d is None:
			mm = self.d.shape[0]-1
		self.slider_select.setMaximum(mm)
		self.slider_select.setValue(self.index)

	def initialize_plots(self):
		[[aaa.cla() for aaa in aa] for aa in self.a]

		lw=.75
		pb=.2
		self.a[0][0].plot(np.random.rand(self.d.shape[0]),color='g',ls=':',alpha=pb,lw=.75)
		self.a[0][0].plot(np.random.rand(self.d.shape[0]),color='g',alpha=.8,lw=.75)
		self.a[0][0].plot(np.random.rand(self.d.shape[0]),color='g',ls=':',alpha=pb,lw=.75)
		self.a[0][0].plot(np.random.rand(self.d.shape[0]),color='r',ls=':',alpha=pb,lw=.75)
		self.a[0][0].plot(np.random.rand(self.d.shape[0]),color='r',alpha=.8,lw=.75)
		self.a[0][0].plot(np.random.rand(self.d.shape[0]),color='r',ls=':',alpha=pb,lw=.75)

		self.a[1][0].plot(np.random.rand(self.d.shape[0]),color='b',ls=':',alpha=pb,lw=.75)
		self.a[1][0].plot(np.random.rand(self.d.shape[0]),color='b',alpha=.8,lw=.75)
		self.a[1][0].plot(np.random.rand(self.d.shape[0]),color='b',ls=':',alpha=pb,lw=.75)

		self.a[0][1].plot(np.random.rand(100),color='g',alpha=.8)
		self.a[0][1].plot(np.random.rand(100),color='r',alpha=.8)
		self.a[1][1].plot(np.random.rand(100),color='b',alpha=.8)

		self.a[0][0].get_shared_y_axes().join(self.a[0][0],self.a[0][1])
		self.a[1][0].get_shared_y_axes().join(self.a[1][0],self.a[1][1])

		self.a[0][0].get_shared_x_axes().join(self.a[0][0],self.a[1][0])
		self.a[0][1].get_shared_x_axes().join(self.a[0][1],self.a[1][1])

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


		self.update()
		self.label_axes()
		self.f.tight_layout()
		self.f.subplots_adjust(hspace=.03,wspace=0.015)
		self.f.canvas.draw()

	def initialize_data(self,data):
		self.d = data

		self.yminmax = np.percentile(self.d.flatten(),[1.,99.])
		self.le_min.setText(str(self.yminmax[0]))
		self.le_max.setText(str(self.yminmax[1]))

		order = self.cross_corr_order()
		self.d = self.d[order]

		q = np.copy(self.d)
		q[:,1] -= self.gui.prefs['bleedthrough']*q[:,0]
		self.fret = q[:,1]/(q[:,0] + q[:,1])
		# self.fret = pb.remove_pb_all(self.fret)
		# cut = np.isnan(self.fret)
		# self.d[:,0][cut] = np.nan
		# self.d[:,0][cut] = np.nan
		self.pre_list = np.zeros(self.d.shape[0],dtype='i')
		l1 = calc_pb_time(self.fret,self.gui.prefs['pb_length'])
		l2 = pb_ensemble(q[:,0] + q[:,1])[1]
		# self.pb_list = np.array([np.min((l1[i],l2[i])) for i in range(l1.size)])
		self.pb_list = l2
		self.class_list = np.zeros(self.d.shape[0])

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

		for f in [tools_cull,tools_cullpb]:
			menu_tools.addAction(f)

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
		self.update_sliders()

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
		self.update_sliders()


	def export_traces(self):
		n = 2 #number of colors
		if not self.d is None:
			dd = self.d.copy()
			if n == 2:
				dd[:,1] -= self.gui.prefs['bleedthrough']*dd[:,0]

			q = np.zeros((dd.shape[0]*n,dd.shape[2]))
			for i in range(n):
				q[i::n] = dd[:,i]

			oname = QFileDialog.getSaveFileName(self, 'Export Traces', '_traces.dat','*.dat')
			if oname[0] != "":
				try:
					np.savetxt(oname[0],q.T,delimiter=',')
				except:
					QMessageBox.critical(self,'Export Traces','There was a problem trying to export the traces')

	def export_classes(self):
		n = 2 #number of colors
		if not self.d is None:
			q = np.zeros((self.d.shape[0],1+2*n),dtype='int')
			q[:,0] = self.class_list
			for i in range(n):
				q[:,1+2*i] = self.pre_list
				q[:,1+2*i+1] = self.pb_list

			oname = QFileDialog.getSaveFileName(self, 'Export Classes/Cuts', '_classes.dat','*.dat')
			if oname[0] != "":
				try:
					np.savetxt(oname[0],q.astype('i'),delimiter=',')
				except:
					QMessageBox.critical(self,'Export Classes','There was a problem trying to export the classes/cuts')

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
				self.update_sliders()

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

	def mouse_click(self,event):
		if (event.inaxes == self.a[0][0] or event.inaxes == self.a[1][0]) and not self.d is None:
			if event.button == 3 and self.toolbar._active is None:
				self.pb_list[self.index] = int(np.round(event.xdata/self.gui.prefs['tau']))
				self.update()
			if event.button == 1 and self.toolbar._active is None:
				self.pre_list[self.index] = int(np.round(event.xdata/self.gui.prefs['tau']))
				self.update()
			if event.button == 2 and self.toolbar._active is None:
				self.pre_list[self.index] = 0
				self.pb_list[self.index] = get_point_pbtime(self.d[self.index].sum(0))
				self.update()
			# x = event.xdata
			# y = event.ydata
			# print x,y,event.button

	def label_axes(self):
		fs = 12
		self.a[0][0].set_ylabel(r'Intensity (a.u.)',fontsize=fs,va='top')
		self.a[1][0].set_ylabel(r'E$_{\rm{FRET}}$',fontsize=fs,va='top')
		self.a[1][0].set_xlabel(r'Time (s)',fontsize=fs)
		self.a[1][1].set_xlabel(r'Probability',fontsize=fs)

	def slide_switch(self,v):
		self.index = v
		self.update()

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

	def make_shortcut(self,key,fxn):
		qs = QShortcut(self)
		qs.setKey(key)
		qs.activated.connect(fxn)

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

	def ensemble_hist(self):
		plt.figure(5)
		f = self.fret
		fpb = f.copy()
		for i in range(f.shape[0]):
			fpb[:self.pre_list[i]] = np.nan
			fpb[self.pb_list[i]:] = np.nan
		plt.hist(f.flatten(),bins=181,range=(-.4,1.4),histtype='stepfilled',alpha=.8,normed=True)
		plt.hist(fpb.flatten(),bins=181,range=(-.4,1.4),histtype='stepfilled',alpha=.8,normed=True)
		plt.show()

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


def launch():
	import sys
	app = QApplication([])
	app.setStyle('fusion')

	g = ui_plotter(None,[])
	app.setWindowIcon(g.windowIcon())
	sys.exit(app.exec_())

if __name__ == '__main__':
	launch()
