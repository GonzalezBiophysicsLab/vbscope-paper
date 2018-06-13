from PyQt5.QtWidgets import QWidget, QSizePolicy,QPushButton,QHBoxLayout,QVBoxLayout,QSlider,QLabel,QSpinBox,QFileDialog,QMessageBox,QFrame
from PyQt5.QtCore import Qt
from PyQt5.Qt import QFont

import cPickle as pickle
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt

_windows = False
import os
if os.name == 'nt':
	_windows = True
import sys

default_prefs = {
	'spotfind_nsearch':3,
	'spotfind_clip_border':8,
	'spotfind_threshold':1e-10,
	'spotfind_maxiterations':2000,
	'spotfind_nstates':8
}


class dock_spotfind(QWidget):
	def __init__(self,parent=None):
		super(dock_spotfind, self).__init__(parent)

		self.default_prefs = default_prefs

		self.flag_priorsloaded = False
		self.flag_spots = False

		self.gui = parent
		self.gmms = None
		self.priors = None
		self.xys = None

		### Menu Widgets
		self.button_toggle = QPushButton('Toggle Spots')
		self.button_export = QPushButton('Export Location')

		label_prior = QLabel('Priors from:')
		self.spin_prior = QSpinBox()
		self.button_loadpriors = QPushButton('Load Priors')
		self.button_savepriors = QPushButton('Save Priors')
		self.button_clearpriors = QPushButton('Clear Priors')

		label_start = QLabel('Start:')
		self.spin_start = QSpinBox()
		label_end = QLabel('End:')
		self.spin_end = QSpinBox()
		self.button_search = QPushButton('Range Find')
		self.button_find = QPushButton('Sum Find')
		self.button_quick = QPushButton('Quick Find')

		self.label_bb = QLabel()
		self.label_pp = QLabel()
		self.label_bb.setFont(QFont('monospace'))
		self.label_pp.setFont(QFont('monospace'))

		self.sliders = [QSlider(Qt.Horizontal)]
		sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
		[s.setSizePolicy(sizePolicy) for s in self.sliders]

		##### Layout
		total_layout = QVBoxLayout()

		wlp = QWidget()
		layout_priors = QHBoxLayout()

		layout_priors.addWidget(label_prior)
		layout_priors.addWidget(self.spin_prior)
		layout_priors.addStretch(1)
		wlp.setLayout(layout_priors)

		wlpp = QWidget()
		layout_priorsp = QHBoxLayout()

		layout_priorsp.addWidget(self.button_savepriors)
		layout_priorsp.addWidget(self.button_loadpriors)
		layout_priorsp.addWidget(self.button_clearpriors)
		layout_priorsp.addStretch(1)
		wlpp.setLayout(layout_priorsp)

		wls1 = QWidget()
		layout_slider1 = QVBoxLayout()
		layout_slider1.addWidget(QLabel('Background Class Threshold'))
		whboxs1 = QWidget()
		hbox_sliders1 = QHBoxLayout()
		self.spins = [QSpinBox() for i in range(4)]
		for sss in self.spins:
			hbox_sliders1.addWidget(sss)
		# hbox_sliders1.addWidget(self.label_bb)
		whboxs1.setLayout(hbox_sliders1)

		layout_slider1.addWidget(whboxs1)
		wls1.setLayout(layout_slider1)

		wls2 = QWidget()
		layout_slider2 = QVBoxLayout()
		layout_slider2.addWidget(QLabel('Spot Probability Threshold'))
		whboxs2 = QWidget()
		hbox_sliders2 = QHBoxLayout()
		hbox_sliders2.addWidget(self.sliders[0])
		hbox_sliders2.addWidget(self.label_pp)
		whboxs2.setLayout(hbox_sliders2)
		layout_slider2.addWidget(whboxs2)
		wls2.setLayout(layout_slider2)

		whf = QWidget()
		hbox_find = QHBoxLayout()
		hbox_find.addWidget(self.button_toggle)
		hbox_find.addWidget(self.button_export)
		whf.setLayout(hbox_find)

		whs = QWidget()
		hbox_search = QHBoxLayout()
		hbox_search.addWidget(label_start)
		hbox_search.addWidget(self.spin_start)
		hbox_search.addWidget(label_end)
		hbox_search.addWidget(self.spin_end)
		whs.setLayout(hbox_search)

		whff = QWidget()
		hbox_ff = QHBoxLayout()
		hbox_ff.addStretch(1)
		hbox_ff.addWidget(self.button_search)
		hbox_ff.addWidget(self.button_find)
		hbox_ff.addWidget(self.button_quick)
		whff.setLayout(hbox_ff)


		total_layout.addWidget(wlp)
		total_layout.addWidget(wlpp)
		sep = QFrame()
		sep.setFrameStyle(4)
		sep.setLineWidth(1)
		total_layout.addWidget(sep)
		total_layout.addWidget(wls1)
		total_layout.addWidget(wls2)
		total_layout.addWidget(whf)
		sep = QFrame()
		sep.setFrameStyle(4)
		sep.setLineWidth(1)
		total_layout.addWidget(sep)
		total_layout.addWidget(whs)
		total_layout.addWidget(whff)
		total_layout.addStretch(1)
		self.setLayout(total_layout)

		#### Initialization
		self.setup_sliders()
		self.connect_things()

	def show_spins(self):
		for i in range(4):
			if i < self.gui.data.ncolors:
				self.spins[i].setVisible(True)
			else:
				self.spins[i].setVisible(False)

	def setup_sliders(self):
		for sss in self.spins:
			sss.setMaximum(100)
			sss.setMinimum(0)
			sss.setValue(0)
		self.show_spins()
		self.sliders[0].setMinimum(0)
		self.sliders[0].setMaximum(1000)

		# Initialize thresholds
		self.sliders[0].setValue(500) # p = .5
		self.change_slide1(self.sliders[0].value())

		[s.setValue(1) for s in [self.spin_prior,self.spin_start]]

		[s.setMinimum(1) for s in [self.spin_prior,self.spin_start,self.spin_end]]
		[s.setMaximum(self.gui.data.total_frames) for s in [self.spin_prior,self.spin_start,self.spin_end]]
		self.spin_end.setValue(self.gui.data.total_frames)

	def connect_things(self):

		self.sliders[0].valueChanged.connect(self.change_slide1)
		[s.sliderReleased.connect(self.update) for s in self.sliders]
		[s.valueChanged.connect(self.update) for s in self.spins]

		self.button_find.clicked.connect(self.findspots)
		self.button_quick.clicked.connect(self.quick_find)
		self.button_search.clicked.connect(self.searchspots)
		self.button_toggle.clicked.connect(self.togglespots)
		self.button_export.clicked.connect(self.exportspots)
		self.button_loadpriors.clicked.connect(self.loadpriors)
		self.button_savepriors.clicked.connect(self.savepriors)
		self.button_clearpriors.clicked.connect(self.clearpriors)

	def flush_old(self):
		self.flag_spots = False
		self.gmms = None
		self.xys = None
		self.locs = None
		self.gui.plot.clear_collections()

	def print_spotnum(self):
		try:
			s = [self.xys[j].shape[-1] for j in range(self.gui.data.ncolors)]
			out = 'Spots found '+str(s)
			self.gui.statusbar.showMessage(out)
		except:
			pass

	def findspots(self):
		if self.gui.data.flag_movie:
			# if not self.flag_priorsloaded:
				# self.get_priors()
			# self.gmms,self.locs = self.setup_spot_find(self.gui.data.current_frame)
			self.gmms,self.locs = self.setup_spot_find()

			if self.gui.prefs['computer_ncpu'] > 1 and not _windows:
				pool = mp.Pool(self.gui.prefs['computer_ncpu'])
				self.gmms = pool.map(_run,self.gmms)
				pool.close()
			else:
				self.gmms = map(_run,self.gmms)
			# for i in range(self.gui.data.ncolors):
			# 	self.gmms[i].run()
			self.update_spots()
			self.gui.plot.clear_collections()
			self.plot_spots()
			self.flag_spots = True
			self.print_spotnum()
			self.gui.docks['contrast'][1].update_image_contrast()
			# self.gui.statusbar.showMessage('Spot finding complete')

	def update_spots(self):
		self.xys = [None for _ in range(self.gui.data.ncolors)]
		self.gui.plot.clear_collections()
		r,shifts = self.gui.data.regions_shifts()
		if len(self.gmms) == self.gui.data.ncolors:
			for i in range(self.gui.data.ncolors):
				v = self.spot_cutoff(self.gmms[i], self.pp, bg_cutoff = self.spins[i].value()).astype('i')
				cut = np.nonzero(v)[0]
				x = self.locs[i][0][cut]
				y = self.locs[i][1][cut]
				self.xys[i] = np.array([x+shifts[i][0],y+shifts[i][1]])
			self.remove_duplicates()
		else:
			n = len(self.gmms)/2
			self.xys = []
			for j in range(self.gui.data.ncolors):
				self.xys.append(np.array(()).reshape((2,0)))
				for i in range(n):
					g = self.gmms[2*i+j]
					l = self.locs[2*i+j]
					v = self.spot_cutoff(g, self.pp, bg_cutoff = self.spins[j].value()).astype('i')
					cut = np.nonzero(v)[0]
					x = l[0][cut]
					y = l[1][cut]
					self.xys[j] = np.append(self.xys[j],np.array((x+shifts[j][0],y+shifts[j][1])),axis=1)
					# self.xys[j][0] = np.append(self.xys[j][0],x+shifts[j][0])
					# self.xys[j][1] = np.append(self.xys[j][1],y+shifts[j][1])
					# np.array([x+shifts[i][0],y+shifts[i][1]])))
			self.compile_spots()


	def searchspots(self):
		if self.gui.data.flag_movie:
			if not self.flag_priorsloaded:
				self.get_priors()

			start = self.spin_start.value() - 1
			end = self.spin_end.value() - 1
			# print start,end
			gmms = []
			locs = []
			for i in range(start,end+1):
				g,l = self.setup_spot_search(i)
				for j in range(self.gui.data.ncolors):
					gmms.append(g[j])
					locs.append(l[j])

			if self.gui.prefs['computer_ncpu'] > 1 and not _windows:
				pool = mp.Pool(self.gui.prefs['computer_ncpu'])
				gmms = pool.map(_run,gmms)
				pool.close()
			else:
				gmms = map(_run,gmms)

			# for g in gmms:
			# 	print g.color,g.frame

			self.gmms = gmms#[-2:]
			self.locs = locs#[-2:]

			self.update_spots()
			self.gui.plot.clear_collections()
			self.plot_spots()
			self.print_spotnum()
			self.flag_spots = True
			# self.gui.statusbar.showMessage('Spot finding complete')

	# def compile_spots(self):
		# self.xys = [None for _ in range(self.gui.data.ncolors)]
		# self.gui.plot.clear_collections()
		# r,shifts = self.gui.data.regions_shifts()
        #
		# n = len(self.gmms)/2
		# total_probs = [np.zeros(self.gui.data.movie[0].shape,dtype='f')[r[j][0][0]:r[j][0][1],r[j][1][0]:r[j][1][1]] for  j in range(self.gui.data.ncolors)]
        #
		# # temp = np.zeros_like(self.gui.data.movie[0],dtype='f')
		# for j in range(self.gui.data.ncolors):
		# 	for i in range(n):
		# 		g = self.gmms[2*i+j]
		# 		l = self.locs[2*i+j]
		# 		class_list = self.not_background_class(g,self.bb)
		# 		probs = (g.r[:,class_list]).sum(1)/g.r.sum(1)
		# 		total_probs[j][l[0],l[1]] += probs
		# 	spots = total_probs[j] > self.pp
		# 	cut = np.nonzero(spots)
		# 	x = cut[0]
		# 	y = cut[1]
		# 	self.xys[j] = np.array([x+shifts[j][0],y+shifts[j][1]])

			# temp[r[j][0][0]:r[j][0][1],r[j][1][0]:r[j][1][1]] = total_probs[j]

		# testing

		# self.gmms = [None for _ in range(self.gui.data.ncolors)]
		# self.gmms = self.gmms[:self.gui.data.ncolors]
		# self.locs = [None for _ in range(self.gui.data.ncolors)]
		# self.locs = self.locs[:self.gui.data.ncolors]
		# self.total_probs = total_probs
		# for j in range(self.gui.data.ncolors):
		# 	# g,l = self.setup_gmm(total_probs[j])
		# 	# g.threshold=1e-10
		# 	# g.run()
		# 	# self.gmms[j] = g
		# 	# self.locs[j] = l
		# 	self.gmms[j].r = total_probs[j]

		# for j in range(self.gui.data.ncolors):


		# self.gui.plot.image.set_array(temp)
		# self.gui.plot.image.set_clim(temp.min(),temp.max())
		# self.gui.plot.canvas.draw()

		# self.gui.plot.clear_collections()
		# self.plot_spots()
		# self.flag_spots = True
		# self.gui.statusbar.showMessage('Spot finding complete')


	def togglespots(self):
		for i in range(len(self.gui.plot.ax.collections)):
			self.gui.plot.ax.collections[i].set_visible(not self.gui.plot.ax.collections[i].get_visible())
		self.gui.plot.canvas.draw()

	def exportspots(self):
		if self.gui.data.flag_movie:
			oname = QFileDialog.getSaveFileName(self, 'Export Spots', self.gui.data.filename[:-4]+'_spots.dat','*.dat')
			if oname[0] != "":
				try:
					for i in range(self.gui.data.ncolors):
						np.savetxt(oname[0][:-4]+'_%d.dat'%(i),np.array(self.xys[i]).T)
				except:
					QMessageBox.critical(self,'Export Status','There was a problem trying to export the spot locations')

	def clearpriors(self):
		self.priors = None
		self.flag_priorsloaded = False

	def loadpriors(self):
		if self.gui.data.flag_movie:
			fname = QFileDialog.getOpenFileName(self,'Choose priors to load','./')#,filter='TIF File (*.tif *.TIF)')
			if fname[0] != "":
				try:
					f = open(fname[0],'r')
					p = pickle.load(f)
					f.close()
					self.priors = p
					self.flag_priorsloaded = True
					self.gui.statusbar.showMessage('Priors loaded from  %s'%(fname[0]))
				except:
					QMessageBox.critical(self,'Load Status','There was a problem trying to load the priors from file %s'%(fname[0]))
					self.flag_priorsloaded = False

	def savepriors(self):
		if self.gui.data.flag_movie:
			oname = QFileDialog.getSaveFileName(self, 'Save Priors', self.gui.data.filename[:-4]+'_priors_%i.dat'%(self.spin_prior.value()),'*.dat')
			if oname[0] != "":
				try:
					## Save Normal
					self.get_priors()
					f = open(oname[0],'w')
					pickle.dump(self.priors,f)
					f.close()
				except:
					QMessageBox.critical(self,'Save Status','There was a problem trying to save the priors of frame %d'%(self.spin_prior.value()-1))

	def plot_spots(self,color=None):
		if not color is None:
			colors = [color for _ in range(self.gui.data.ncolors)]
		else:
			colors = self.gui.prefs['channels_colors']
		if not self.xys is None:
			for i in range(self.gui.data.ncolors):
				self.gui.plot.scatter(self.xys[i][0],self.xys[i][1],color=colors[i])
			self.gui.plot.canvas.draw()

	def change_slide1(self,p):
		self.pp = float(p)/1000.
		self.label_pp.setText('%.4f'%self.pp)
		# if not self.gmms is None:
			# self.update_spots()

	# def not_background_class(self,gmm,cutoff=.001):
	# 	l = np.arange(gmm.post.m.size)
	# 	# if gmm._bg_flag:
	# 	# 	p_bg = np.exp(gmm.background.lnprob(gmm.post.m)) # prob of being max-val background
	# 	# 	p = np.exp(gmm.background.lnprob(gmm.background.e_max_m))
	# 	# 	cut = (gmm.post.m > gmm.background.e_max_m)*(p_bg < p*cutoff)
	# 	# 	return l[cut]
	# 	l = l[:]
	#
	# 	return l

	def spot_cutoff(self,gmm,p_cutoff,bg_cutoff=0,class_list=None):
		if class_list is None:
			# class_list = self.not_background_class(gmm,bg_cutoff)
			class_list = np.arange(gmm.post.m.size)
			class_list = class_list[class_list >= bg_cutoff]
			print class_list,gmm.r.shape

		probs = (gmm.r[:,class_list]).sum(1)/gmm.r.sum(1)
		spots = probs > p_cutoff
		return spots

	def update(self):
		if self.gui.data.flag_movie and not self.gmms is None:
			self.update_spots()
			self.plot_spots()
			self.print_spotnum()

	def setup_gmm(self,dd,prior=None,max_frames = 1):
		p = self.gui.prefs

		#bg = self.gui.docks['background'][1].calc_background(dd)
		image = dd#-bg

		## Find local mins and local maxes
		from ..supporting import minmax
		mmin,mmax = minmax.minmax_map(image,p['spotfind_nsearch'],p['spotfind_clip_border'])

		## Estimate background distribution from local mins

		if not 'src.supporting.normal_minmax_dist' in sys.modules:
			self.gui.set_status('Compiling...')
		from ..supporting import normal_minmax_dist as nd
		from ..supporting import vbem_gmm as vb
		self.gui.set_status('Finding spots...')
		bgfit = nd.estimate_from_min(image[mmin],p['spotfind_nsearch']**2 * max_frames)
		background = vb.background(p['spotfind_nsearch']**2 * max_frames,*bgfit)

		## Classify local maxes
		if not prior is None:
			gmm = vb.vbem_gmm(image[mmax], p['spotfind_nstates'], bg=background, prior=prior)
		else:
			gmm = vb.vbem_gmm(image[mmax], p['spotfind_nstates'], bg=background)
		gmm.threshold = p['spotfind_threshold']
		gmm.maxiters = p['spotfind_maxiterations']
		gmm._debug = False

		locs = np.nonzero(mmax)

		return gmm,locs

	def get_priors(self):
		self.gui.statusbar.showMessage('Finding priors from frame %d'%(self.spin_prior.value()))
		self.gui.app.processEvents()

		nc = self.gui.data.ncolors
		dp = self.gui.data.movie[self.spin_prior.value()-1].astype('f')

		regions,shifts = self.gui.data.regions_shifts()

		self.priors = [None for _ in range(nc)]
		for i in range(nc):
			r = regions[i]
			ddp = dp[r[0][0]:r[0][1],r[1][0]:r[1][1]]
			gmm,locs = self.setup_gmm(ddp,prior=None)
			self.priors[i] = gmm.prior

	# def setup_spot_find(self,frame):
	def setup_spot_find(self):
		nc = self.gui.data.ncolors
		# d = self.gui.data.movie[frame].astype('f')

		start = self.spin_start.value() - 1
		end = self.spin_end.value() - 1
		total = end + 1 - start
		# d = np.max(self.gui.data.movie[start:end+1],axis=0).astype('f')
		d = np.sum(self.gui.data.movie[start:end+1],axis=0).astype('f') / float(total)
		# dmin = np.min(self.gui.data.movie[start:end+1],axis=0).astype('f')

		bg = self.gui.docks['background'][1].calc_background(d)
		self.disp_image = d-bg
		self.gui.plot.image.set_array(self.disp_image)
		self.gui.docks['contrast'][1].update_image_contrast()

		regions,shifts = self.gui.data.regions_shifts()
		locs = [None for _ in range(nc)]
		gmms = [None for _ in range(nc)]

		self.gui.statusbar.showMessage('Finding spots...')
		self.gui.app.processEvents()

		self.gui.prefs['spotfind_nsearch']
		for i in range(nc):
			r = regions[i]
			# dd = d[r[0][0]:r[0][1],r[1][0]:r[1][1]]
			dd = (d-bg)[r[0][0]:r[0][1],r[1][0]:r[1][1]]

			# gmm,loc = self.setup_gmm(dd,self.priors[i])
			gmm,loc = self.setup_gmm(dd,prior=None)
			# gmm.frame = frame
			gmm.color = i

			locs[i] = loc
			gmms[i] = gmm
		return gmms,locs


	# def setup_spot_find(self,frame):
	def setup_spot_search(self,frame):
		nc = self.gui.data.ncolors
		d = self.gui.data.movie[frame].astype('f')

		# start = self.spin_start.value() - 1
		# end = self.spin_end.value() - 1
		# total = end + 1 - start
		# d = np.max(self.gui.data.movie[start:end+1],axis=0).astype('f')
		# d = np.sum(self.gui.data.movie[start:end+1],axis=0).astype('f') / float(total)
		# dmin = np.min(self.gui.data.movie[start:end+1],axis=0).astype('f')

		bg = self.gui.docks['background'][1].calc_background(d)
		# self.disp_image = d#-bg
		# self.gui.plot.image.set_array(self.disp_image)
		# self.gui.docks['contrast'][1].update_image_contrast()

		regions,shifts = self.gui.data.regions_shifts()
		locs = [None for _ in range(nc)]
		gmms = [None for _ in range(nc)]

		self.gui.statusbar.showMessage('Finding spots...')
		self.gui.app.processEvents()

		self.gui.prefs['spotfind_nsearch']
		for i in range(nc):
			r = regions[i]
			# dd = d[r[0][0]:r[0][1],r[1][0]:r[1][1]]
			dd = (d-bg)[r[0][0]:r[0][1],r[1][0]:r[1][1]]

			gmm,loc = self.setup_gmm(dd,self.priors[i])
			# gmm,loc = self.setup_gmm(dd,prior=None)
			gmm.frame = frame
			gmm.color = i

			locs[i] = loc
			gmms[i] = gmm
		return gmms,locs

	def compile_spots(self):
		nc = self.gui.data.ncolors
		regions,shifts = self.gui.data.regions_shifts()
		p = self.gui.prefs
		for i in range(nc):
			mm = make_map(self.xys[i],self.gui.data.movie.shape[1],self.gui.data.movie.shape[2])


			from ..supporting import minmax
			r = regions[i]
			mmm = mm[r[0][0]:r[0][1],r[1][0]:r[1][1]]
			mmin,mmax = minmax.minmax_map(mmm,p['spotfind_nsearch'],p['spotfind_clip_border'])

			from ..supporting import normal_minmax_dist as nmd
			bg = nmd.estimate_from_min(mmm[mmin],p['spotfind_nsearch']**2)

			cut = np.max((1.,bg[0] + np.sqrt(bg[1])*3.)) # + np.sqrt(bg[0])*3.
			print i,cut

			mmin,mmax = minmax.minmax_map(mm*(mm > cut),p['spotfind_nsearch'],p['spotfind_clip_border'])
			self.xys[i] = np.array(np.nonzero(mmax))

			if i == 0:
				mmtotal = np.zeros_like(mm)
			# mmtotal[mm>cut] += mm[mm>cut]
			mmtotal += mm

		self.disp_image = mmtotal
		self.gui.plot.image.set_array(self.disp_image)
		self.gui.docks['contrast'][1].update_image_contrast()


	def remove_duplicates(self):
		for i in range(self.gui.data.ncolors):
			# self.xys[i] = cull_rep_px(self.xys[i].astype('double'),self.gui.data.movie.shape[1],self.gui.data.movie.shape[2])
			self.xys[i] = avg_close(self.xys[i].astype('double'),self.gui.prefs['extract_same_cutoff'])

	def quick_find(self):
		if self.gui.data.flag_movie:
			self.xys = [None for _ in range(self.gui.data.ncolors)]
			self.gui.plot.clear_collections()
			nc = self.gui.data.ncolors
			start = self.spin_start.value() - 1
			end = self.spin_end.value() - 1
			total = end + 1 - start
			d = self.gui.data.movie[start:end+1]

			regions,shifts = self.gui.data.regions_shifts()

			p = self.gui.prefs
			from ..supporting import minmax
			from ..supporting import normal_minmax_dist as nd
			self.disp_image = np.zeros_like(self.gui.data.movie[0],dtype='float32')
			for i in range(nc):
				r = regions[i]
				d = self.gui.data.movie[start:end+1,r[0][0]:r[0][1],r[1][0]:r[1][1]].astype('f')
				bg = self.gui.docks['background'][1].calc_background(d)
				bg = bg.astype('f')
				if bg.ndim == 3:
					bg = np.median(bg,axis=2)
				d = d.copy()/bg[:,:,None]

				####
				bg = np.min(d,axis=0)
				mmin,mmax = minmax.minmax_map(bg,p['spotfind_nsearch'],p['spotfind_clip_border'])
				bgfit = nd.estimate_from_min(bg[mmin],p['spotfind_nsearch']**2 * d.shape[0])
				print i,bgfit
				# prob = np.sum(self.quick_mix_ll(d,bgfit[0],bgfit[1]) > self.pp,axis=0)
				prob = np.mean(self.quick_mix_ll(d,bgfit[0],bgfit[1]) ,axis=0)#> self.pp
				prob = minmax.clip(prob,p['spotfind_clip_border'])
				self.disp_image[r[0][0]:r[0][1],r[1][0]:r[1][1]] += prob

				mmin,mmax = minmax.minmax_map(prob,p['spotfind_nsearch'],p['spotfind_clip_border'])
				bgfit = nd.estimate_from_min(prob[mmin],(p['spotfind_nsearch'])**2)
				print i,bgfit
				prob = self.quick_mix_ll(prob,bgfit[0],bgfit[1])
				prob = minmax.clip(prob,p['spotfind_clip_border'])
				# self.disp_image[r[0][0]:r[0][1],r[1][0]:r[1][1]] += prob
				# mmin,mmax = minmax.minmax_map(prob,p['spotfind_nsearch'],p['spotfind_clip_border'])

				from scipy.ndimage import label, center_of_mass
				labels,num = label(prob > self.pp)
				coms = np.array(center_of_mass(prob,labels,range(1,num+1))).T
				self.xys[i] = coms
				###
				# self.xys[i] = np.array(np.nonzero(prob*mmax > self.pp))
				self.xys[i][0] += shifts[i][0]
				self.xys[i][1] += shifts[i][1]

			self.gui.plot.image.set_array(self.disp_image)
			self.gui.docks['contrast'][1].update_image_contrast()

			self.gui.plot.clear_collections()
			self.plot_spots()
			self.flag_spots = True
			self.print_spotnum()
			self.gui.docks['contrast'][1].update_image_contrast()
			# self.gui.statusbar.showMessage('Spot finding complete')

	def quick_mix_ll(self,d,m,v):
		pnormal = 1./np.sqrt(2.*np.pi*v)*np.exp(-.5*(d-m)**2. / v)
		puniform_high = 1./(d.max() - m)*(d > m) ## assume 16 bit camera
		puniform_low = 1./(m - d.min())*(d<=m) ## assume 16 bit camera
		p = puniform_high/(pnormal+puniform_high+puniform_low)
		p[np.bitwise_not(np.isfinite(p))] = 0.
		return p


import numba as nb
@nb.jit(["int64[:,:](double[:,:],int64,int64)","int64[:,:](int64[:,:],int64,int64)"],nopython=True)
def make_map(ss,nx,ny):
	x = ss[0]
	y = ss[1]

	m = np.zeros((nx,ny),dtype=nb.int64)
	for i in range(x.size):
		m[int(x[i]),int(y[i])] += 1
	return m

# import numba as nb
# @nb.jit(["double[:,:](double[:,:],int64,int64)","int64[:,:](int64[:,:],int64,int64)"],nopython=True)
# def cull_rep_px(ss,nx,ny):
# 	x = ss[0]
# 	y = ss[1]
#
# 	m = np.zeros((nx,ny))
# 	for i in range(x.size):
# 		m[int(x[i]),int(y[i])] += 1
#
# 	x = []
# 	y = []
# 	for i in range(nx):
# 		for j in range(ny):
# 			if m[i,j] > 0:
# 				x.append(i)
# 				y.append(j)
# 	return np.array((x,y),dtype=ss.dtype)
#
@nb.jit('double[:,:](double[:,:],double)',nopython=True)
def avg_close(ss,cutoff):

	totalx = []
	totaly = []

	# ## AVERAGE
	# already = []
	# for j in range(ss[0].size):
	# 	currentn = 1.
	# 	currentx = ss[0][j]
	# 	currenty = ss[1][j]
	# 	if already.count(j) == 0:
	# 		for i in range(j+1,ss[0].size):
	# 			if already.count(i) == 0:
	# 				r = np.sqrt((ss[0][i] - ss[0][j])**2. + (ss[1][i] - ss[1][j])**2.)
	# 				if r < cutoff:
	# 					currentx += ss[0][i]
	# 					currenty += ss[1][i]
	# 					currentn += 1.
	# 					already.append(i)
	# 		totalx.append(currentx/currentn)
	# 		totaly.append(currenty/currentn)
	# 		already.append(j)
	#### FIRST
	already = []
	for j in range(ss[0].size):
		# currentn = 1.
		# currentx = ss[0][j]
		# currenty = ss[1][j]
		if already.count(j) == 0:
			for i in range(j+1,ss[0].size):
				if already.count(i) == 0:
					r = np.sqrt((ss[0][i] - ss[0][j])**2. + (ss[1][i] - ss[1][j])**2.)
					if r < cutoff:
						# currentx += ss[0][i]
						# currenty += ss[1][i]
						# currentn += 1.
						already.append(i)

			# totalx.append(currentx/currentn)
			# totaly.append(currenty/currentn)
			totalx.append(ss[0][j])
			totaly.append(ss[1][j])
			already.append(j)

	return np.array((totalx,totaly))


def _run(a):
	a.run()
	return a
