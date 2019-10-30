from PyQt5.QtWidgets import QWidget, QSizePolicy,QPushButton,QHBoxLayout,QVBoxLayout,QSlider,QLabel,QSpinBox,QFileDialog,QMessageBox,QFrame,QInputDialog
from PyQt5.QtCore import Qt
from PyQt5.Qt import QFont

import pickle as pickle
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
import time

_windows = False
import os
if os.name == 'nt':
	_windows = True
import sys

default_prefs = {
	'spotfind_nsearch':3,
	'spotfind_clip_border':8,
	'spotfind_threshold':1e-8,
	'spotfind_maxiterations':250,
	'spotfind_nstates':1,
	'spotfind_frameson':5,
	'spotfind_nrestarts':4,
	'spotfind_rangefast':False,
	'spotfind_updatebg':True,
	'spotfind_acfn':1,
	'spotfind_acfstd':0.1,
}


class dock_spotfind(QWidget):
	parallel_total = 0

	def __init__(self,parent=None):
		super(dock_spotfind, self).__init__(parent)

		self.default_prefs = default_prefs

		self.flag_priorsloaded = False
		self.flag_spots = False
		self.flag_importedvb = False
		self.flag_importedvbmax = False

		self.gui = parent
		# self.gmms = None
		self.xys = None
		self.spotprobs = None

		### Menu Widgets
		self.button_toggle = QPushButton('Toggle Spots')
		self.button_export = QPushButton('Export Location')

		label_prior = QLabel('Priors from:')
		# self.spin_prior = QSpinBox()
		self.button_loadpriors = QPushButton('Load Priors')
		self.button_savepriors = QPushButton('Save Priors')
		self.button_clearpriors = QPushButton('Clear Priors')

		label_start = QLabel('Start:')
		self.spin_start = QSpinBox()
		label_end = QLabel('End:')
		self.spin_end = QSpinBox()
		# self.button_search = QPushButton('Range Find')
		# self.button_find = QPushButton('Sum Find')
		# self.button_quick = QPushButton('Quick Find')
		self.button_threshold = QPushButton('ACF(t=n)')
		self.button_vbmax = QPushButton('Mean')
		self.button_range = QPushButton('Range')

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
		# layout_priors.addWidget(self.spin_prior)
		# layout_priors.addStretch(1)
		wlp.setLayout(layout_priors)

		wlpp = QWidget()
		layout_priorsp = QHBoxLayout()

		layout_priorsp.addWidget(self.button_savepriors)
		layout_priorsp.addWidget(self.button_loadpriors)
		layout_priorsp.addWidget(self.button_clearpriors)
		# layout_priorsp.addStretch(1)
		wlpp.setLayout(layout_priorsp)

		# wls1 = QWidget()
		# layout_slider1 = QVBoxLayout()
		# layout_slider1.addWidget(QLabel('Background Class Threshold'))
		# whboxs1 = QWidget()
		# hbox_sliders1 = QHBoxLayout()
		# self.spins = [QSpinBox() for i in range(4)]
		# for sss in self.spins:
		# 	hbox_sliders1.addWidget(sss)
		# # hbox_sliders1.addWidget(self.label_bb)
		# whboxs1.setLayout(hbox_sliders1)
		#
		# layout_slider1.addWidget(whboxs1)
		# wls1.setLayout(layout_slider1)

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
		# hbox_ff.addStretch(1)
		hbox_ff.addWidget(self.button_threshold)
		hbox_ff.addWidget(self.button_range)
		hbox_ff.addWidget(self.button_vbmax)

		whff.setLayout(hbox_ff)

		total_layout.addStretch(1)
		total_layout.addWidget(wlp)
		total_layout.addWidget(wlpp)
		sep = QFrame()
		sep.setFrameStyle(4)
		sep.setLineWidth(1)
		total_layout.addWidget(sep)
		# total_layout.addWidget(wls1)
		total_layout.addWidget(wls2)
		total_layout.addWidget(whf)
		sep = QFrame()
		sep.setFrameStyle(4)
		sep.setLineWidth(1)
		total_layout.addWidget(sep)
		total_layout.addWidget(whs)
		total_layout.addWidget(whff)

		self.setLayout(total_layout)

		#### Initialization
		self.setup_sliders()
		self.connect_things()

	# def show_spins(self):
	# 	for i in range(4):
	# 		if i < self.gui.data.ncolors:
	# 			self.spins[i].setVisible(True)
	# 		else:
	# 			self.spins[i].setVisible(False)

	def setup_sliders(self):
		# for sss in self.spins:
		# 	sss.setMaximum(100)
		# 	sss.setMinimum(0)
		# 	sss.setValue(0)
		# self.show_spins()
		self.sliders[0].setMinimum(0)
		self.sliders[0].setMaximum(1000)

		# Initialize thresholds
		self.sliders[0].setValue(500) # p = .5
		self.change_slide1()

		[s.setValue(1) for s in [self.spin_start]]
		[s.setMinimum(1) for s in [self.spin_start,self.spin_end]]
		[s.setMaximum(self.gui.data.total_frames) for s in [self.spin_start,self.spin_end]]
		# [s.setValue(1) for s in [self.spin_prior,self.spin_start]]
		# [s.setMinimum(1) for s in [self.spin_prior,self.spin_start,self.spin_end]]
		# [s.setMaximum(self.gui.data.total_frames) for s in [self.spin_prior,self.spin_start,self.spin_end]]
		self.spin_end.setValue(self.gui.data.total_frames)

	def connect_things(self):

		self.sliders[0].sliderReleased.connect(self.change_slide1)
		[s.sliderReleased.connect(self.update) for s in self.sliders]
		# [s.valueChanged.connect(self.update) for s in self.spins]

		self.button_threshold.clicked.connect(self.threshold_find)
		self.button_vbmax.clicked.connect(self.vb_maxval_gmm_find)
		self.button_range.clicked.connect(self.vbscope_range)

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



	def togglespots(self):
		for i in range(len(self.gui.plot.ax.collections)):
			self.gui.plot.ax.collections[i].set_visible(not self.gui.plot.ax.collections[i].get_visible())
		self.gui.plot.canvas.draw()

	def exportspots(self):
		if self.gui.data.flag_movie:
			# oname = QFileDialog.getSaveFileName(self, 'Export Spots', self.gui.data.filename[:-4]+'_spots.dat','*.dat')

			try:
				import os
				from PyQt5.QtCore import QFileInfo
				defaultname = QFileInfo(self.gui.data.filename)
				path = self.gui.latest_directory + os.sep
				newfilename = path + defaultname.fileName().split('.')[0]+'_spots.dat'
				oname = QFileDialog.getSaveFileName(self.gui, 'Export Spots', newfilename,'*.dat')
				if oname[0] != "" and oname[1] != '':
					self.gui.latest_directory = QFileInfo(oname[0]).path()
			except:
				oname = QFileDialog.getSaveFileName(self.gui, 'Export Spots', './','*.dat')


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
			from PyQt5.QtCore import QFileInfo
			fname = QFileDialog.getOpenFileName(self,'Choose priors to load',self.gui.latest_directory)
			if fname[0] != "" and fname[1] != '':
				self.gui.latest_directory = QFileInfo(fname[0]).path()

			if fname[0] != "":
				try:
					self.priors = np.loadtxt(fname[0]).astype('double')
					self.flag_priorsloaded = True
					self.gui.statusbar.showMessage('Priors loaded from  %s'%(fname[0]))
				except:
					QMessageBox.critical(self,'Load Status','There was a problem trying to load the priors')
					self.flag_priorsloaded = False

	def savepriors(self):
		if self.gui.data.flag_movie:
			try:
				import os
				from PyQt5.QtCore import QFileInfo
				defaultname = QFileInfo(self.gui.data.filename)
				path = self.gui.latest_directory + os.sep
				newfilename = path + defaultname.fileName().split('.')[0]+'_priors.dat'
				oname = QFileDialog.getSaveFileName(self.gui, 'Save Priors', newfilename,'*.dat')
				if oname[0] != "" and oname[1] != '':
					self.gui.latest_directory = QFileInfo(oname[0]).path()
			except:
				oname = QFileDialog.getSaveFileName(self.gui, 'Save Priors', './','*.dat')

			if oname[0] != "":
				if not self.priors is None:
					try:
						np.savetxt(oname[0],self.priors)
					except:
						QMessageBox.critical(self,'Save Status','There was a problem trying to save the priors')

	def plot_spots(self,color=None):
		if not color is None:
			colors = [color for _ in range(self.gui.data.ncolors)]
		else:
			colors = self.gui.prefs['channels_colors']
		if not self.xys is None:
			for i in range(self.gui.data.ncolors):
				if not self.xys[i] is None:
					self.gui.plot.scatter(self.xys[i][0],self.xys[i][1],color=colors[i])
			self.gui.plot.canvas.draw()

	def change_slide1(self):
		self.pp = float(self.sliders[0].value())/1000.
		self.label_pp.setText('%.4f'%self.pp)
		self.update_spots()
		# if not self.gmms is None:
			# self.update_spots()

	def update(self):
		try:
			self.update_spots()
		except:
			pass

	def remove_duplicates(self):
		for i in range(self.gui.data.ncolors):
			# self.xys[i] = cull_rep_px(self.xys[i].astype('double'),self.gui.data.movie.shape[1],self.gui.data.movie.shape[2])
			self.xys[i] = avg_close(self.xys[i].astype('double'),self.gui.prefs['extract_same_cutoff'])

	def update_spots(self):
		if not self.spotprobs is None:
			self.gui.plot.clear_collections()
			nc = self.gui.data.ncolors
			for i in range(nc):
				if not self.spotprobs[i] is None:
					if self.spotprobs[i].ndim == 2:
						pmap = self.spotprobs[i] > self.pp
					elif self.spotprobs[i].ndim == 3:
						pmap = self.spotprobs[i].copy()
						pmap *= (pmap > self.pp)
						pmap = self.compile_map(pmap) > self.pp
					x,y = np.nonzero(pmap)
					if x.size > 0:
						self.xys[i] = np.array([x + self.shifts[i][0],y + self.shifts[i][1]])
					else:
						self.xys[i] = None

			self.plot_spots()
			self.flag_spots = True
			self.print_spotnum()
			self.gui.app.processEvents()
			# self.gui.docks['contrast'][1].update_image_contrast()

	def cancel_expt(self):
		self.flag_abort = True

	def estimate_prior_set(self,d):
		strength = np.max((1.,d.size/1000.))
		# strength = 100.
		priors = np.array((strength,.5*strength,.5*np.sum((d-d.mean())**2.),strength),dtype='double')
		return priors

	def priors_check(self,nc):
		if not self.flag_priorsloaded:
			self.priors = np.zeros((nc,4))
		else:
			if self.priors.shape[0] != nc:
				self.priors = np.zeros((nc,4))


	def vbscope_range(self):
		self.gui.set_status('Compiling...')
		from ..supporting.vbmax_em_gmm import vbmax_em_gmm as gmm
		from ..supporting import normal_minmax_dist as nmd
		from ..supporting import minmax as minmax
		self.gui.set_status('Running...')
		self.gui.app.processEvents()

		if not self.gui.data.flag_movie:
			return
		nc = self.gui.data.ncolors
		self.xys = [None for _ in range(nc)]
		self.spotprobs = [None for _ in range(nc)]
		self.gui.plot.clear_collections()

		start = self.spin_start.value() - 1
		end = self.spin_end.value() - 1
		total = end + 1 - start

		self.priors_check(nc)
		regions,self.shifts = self.gui.data.regions_shifts()

		p = self.gui.prefs
		nregion = self.gui.prefs['spotfind_nsearch']**2
		nxy = (p['spotfind_nsearch']-1)//2
		nt = 0

		self.disp_image = np.zeros_like(self.gui.data.movie[0]).astype('double')
		self.spotprobs = [None for _ in range(nc)]

		for i in range(nc):
			self.gui.set_status('Finding spots - %d'%(i))
			self.gui.app.processEvents()

			r = regions[i]
			probs =  np.zeros((total,*self.disp_image.shape))[:,r[0][0]:r[0][1],r[1][0]:r[1][1]]
			self.todo = [None,]*(end+1-start)

			for t in range(start,end+1):
				# if self.flag_abort:
					# break

				d = self.gui.docks['background'][1].bg_filter(self.gui.data.movie[t, r[0][0]:r[0][1], r[1][0]:r[1][1]].astype('double'))

				if not self.flag_priorsloaded and t == start:
					self.priors[i] = self.estimate_prior_set(d)

				if (p['spotfind_rangefast'] and t == start) or (not p['spotfind_rangefast']):
					mmin,mmax = minmax.minmax_map(d[None,:,:], nt,nxy,nxy, p['spotfind_clip_border'])

				h0 = d[mmax[0]].astype('double')
				l0 = d[mmin[0]]

				if (p['spotfind_rangefast'] and t == start) or (not p['spotfind_rangefast']):
					bg_values = nmd.estimate_from_min(l0,nregion)

				self.todo[t-start] = [gmm, t,t-start, i, mmax[0], h0, bg_values, nregion, p['spotfind_nstates'], p['spotfind_maxiterations'],p['spotfind_threshold'],self.priors,self.gui.prefs['spotfind_updatebg']]

				self.disp_image[r[0][0]:r[0][1],r[1][0]:r[1][1]] += d/float(total)

			self.setup_todo_prog(i,start,end)
			with mp.Pool(processes=p['computer_ncpu']) as pool:
				# out = pool.map_async(run_todo,todo)
				results = [pool.apply_async(run_todo,args=(tt,),callback=self.receive_prog) for tt in self.todo]
				for result in results:
					if self.flag_abort:
						del pool
						self.cleanup_progress()
						return None
					result.wait()
					if self.parallel_total > 0:
					# self.parallel_total%1 == 0:
						t1 = time.clock()
						self.todo_prog.setLabelText('Color %d, %d - %d\nCurrent: %d - %.4f sec'%(self.todo_prog_data[0],self.todo_prog_data[1]+1,self.todo_prog_data[2]+1,self.parallel_total +2, (t1-self.todo_t0)/self.parallel_total))
						self.todo_prog.setValue(self.parallel_total)
						self.gui.app.processEvents()

			for result in self.todo:
				if result[0]:
					probs[result[2]][result[3]] = result[4]
			self.cleanup_progress()

				# results = [pool.apply_async(run_todo,args=(tt,self.flag_abort),callback=self.update_todo_prog) for tt in self.todo]
				# pool.close()
				# pool.join()
			# self.todo = [result.get() for result in results]

			# 	results = [p.get() for p in results]
			# 	pool.close()
			# out = map(self.run_todo,todo)
			# for result in self.todo:
				# if result[0]:
					# probs[result[2]][result[3]] = result[4]
			# self.cleanup_progress()

			self.spotprobs[i] = probs

		self.gui.plot.image.set_array(self.disp_image)
		self.update_spots()

	def setup_todo_prog(self,i,start,end):
		from ..ui.ui_progressbar import progressbar
		self.todo_prog = progressbar()
		self.todo_prog.setRange(start,end)
		self.todo_prog.setWindowTitle("Finding Spots")
		self.todo_prog.setLabelText('Setting up color %d'%(i))
		self.todo_prog.canceled.connect(self.cancel_expt)
		self.todo_prog.show()
		self.gui.app.processEvents()

		self.todo_prog_data = [i,start,end]
		self.parallel_total = 0

		self.flag_abort = False
		self.todo_t0 = time.clock()

	def receive_prog(self,result):
		self.parallel_total += 1
		self.todo[result[2]] = result

	def cleanup_progress(self):
		self.todo_prog.close()
		self.flag_abort = True
		self.todo_prog = None

	def compile_map(self,probs):
		from ..supporting import minmax as minmax
		nxy = (self.gui.prefs['spotfind_nsearch']-1)//2

		## Compile maps into one map
		pmap0 = probs.astype('double').mean(0)
		mmin,mmax = minmax.minmax_map(pmap0.reshape((1,pmap0.shape[0],pmap0.shape[1])),0,nxy,nxy,self.gui.prefs['spotfind_clip_border'])
		pmap = pmap0*mmax[0] ## Only use local max!

		## Use model selection with something never on (M1) vs something on frameson (M2)
		from scipy.special import betaln
		a1 = 1.
		b1 = float(probs.shape[0]) + 1.
		m1 = (a1 - 1.)*np.log(pmap)
		m1 += (b1 - 1.)*np.log(1.-pmap+1e-30)
		m1 -= betaln(a1,b1)

		a2 = float(self.gui.prefs['spotfind_frameson']) + 1.
		b2 = float(probs.shape[0]) - float(self.gui.prefs['spotfind_frameson']) + 1.
		m2 = (a2 - 1.)*np.log(pmap)
		m2 += (b2 - 1.)*np.log(1.-pmap+1e-30)
		m2 -= betaln(a2,b2)

		pmap = 1./(np.exp(m1 - m2) + 1.)
		pmap[np.isnan(pmap)] = 0.
		return pmap

	def vb_maxval_gmm_find(self):
		self.gui.set_status('Compiling...')
		from ..supporting.vbmax_em_gmm import vbmax_em_gmm as gmm
		from ..supporting import normal_minmax_dist as nmd
		from ..supporting import minmax as minmax
		self.gui.set_status('Running...')
		self.gui.app.processEvents()

		if not self.gui.data.flag_movie:
			return
		nc = self.gui.data.ncolors
		self.xys = [None for _ in range(nc)]
		self.spotprobs = [None for _ in range(nc)]
		self.gui.plot.clear_collections()

		start = self.spin_start.value() - 1
		end = self.spin_end.value() - 1
		total = end + 1 - start

		self.priors_check(nc)
		regions,self.shifts = self.gui.data.regions_shifts()

		p = self.gui.prefs
		nregion = self.gui.prefs['spotfind_nsearch']**2
		nxy = (p['spotfind_nsearch']-1)//2
		nt = 0

		self.disp_image = np.zeros_like(self.gui.data.movie[0]).astype('double')

		for i in range(nc):
			self.gui.set_status('Finding spots - %d'%(i))
			self.gui.app.processEvents()

			r = regions[i]
			d = self.gui.docks['background'][1].bg_filter(self.gui.data.movie[start:end+1,r[0][0]:r[0][1],r[1][0]:r[1][1]].astype('double').mean(0))

			mmin,mmax = minmax.minmax_map(d[None,:,:],nt,nxy,nxy,p['spotfind_clip_border'])

			h0 = d[mmax[0]].astype('double')
			l0 = d[mmin[0]]

			if not self.flag_priorsloaded:
				self.priors[i] = self.estimate_prior_set(d)

			bg_values = nmd.estimate_from_min(l0,nregion)

			out = gmm(h0, self.gui.prefs['spotfind_nstates'], bg_values, nregion, initials=None, maxiters=self.gui.prefs['spotfind_maxiterations'], threshold=self.gui.prefs['spotfind_threshold'], prior_strengths=self.priors[i],update_bg=self.gui.prefs['spotfind_updatebg'])

			prob = np.zeros_like(d)
			prob[mmax[0]] = (1.-out.r[:,0])

			self.disp_image[r[0][0]:r[0][1],r[1][0]:r[1][1]] += d
			self.spotprobs[i] = prob

		if not self.flag_priorsloaded:
			self.flag_priorsloaded = True

		self.gui.plot.image.set_array(self.disp_image)
		self.gui.docks['contrast'][1].update_image_contrast()
		self.gui.set_status('Finished')
		self.gui.app.processEvents()
		self.update_spots()


	def threshold_find(self):
		if self.gui.data.flag_movie:
			self.xys = [None for _ in range(self.gui.data.ncolors)]
			self.spotprobs = [None for _ in range(self.gui.data.ncolors)]
			self.gui.plot.clear_collections()
			nc = self.gui.data.ncolors
			start = self.spin_start.value() - 1
			end = self.spin_end.value() - 1
			total = end + 1 - start
			d = self.gui.data.movie[start:end+1]

			regions,self.shifts = self.gui.data.regions_shifts()

			p = self.gui.prefs
			from ..supporting import minmax
			# from ..supporting import normal_minmax_dist as nmd
			self.disp_image = np.zeros_like(self.gui.data.movie[0],dtype='float32')

			nxy = (p['spotfind_nsearch']-1)//2
			acfn = p['spotfind_acfn']
			self.disp_image = np.zeros_like(self.gui.data.movie[0],dtype='float32')
			for i in range(nc):
				r = regions[i]
				# d = self.gui.data.movie[start:end+1,r[0][0]:r[0][1],r[1][0]:r[1][1]].astype('f').mean(0)
				# d = self.gui.docks['background'][1].bg_filter(d)
				d = self.gui.data.movie[start:end+1,r[0][0]:r[0][1],r[1][0]:r[1][1]].astype('f')
				d -= d.mean(0)[None,:,:]
				d = np.mean(d[acfn:]*d[:-acfn],axis=0)/np.mean(d**2.,axis=0) ## ACF(t=1)
				# d = self.gui.docks['background'][1].bg_filter(d)
				import scipy.ndimage as nd
				d = nd.gaussian_filter(d,.5)

				# d[d<=0] = 0.
				mmin,mmax = minmax.minmax_map(d[None,:,:],0,nxy,nxy,p['spotfind_clip_border'])
				nregion = (2*nxy+1)**2

				bg_values = np.array((0.,p['spotfind_acfstd']**2.))

				prob = np.zeros_like(d)
				prob[mmax[0]] = calc_unimix_map(d[mmax[0]],bg_values)
				# prob[mmax[0]] = d[mmax[0]]#calc_unimix_map(d[mmax[0]],bg_values)

				# prob = np.zeros_like(d) + d.min()
				# prob[mmax[0]] = d[mmax[0]]
				self.disp_image[r[0][0]:r[0][1],r[1][0]:r[1][1]] += d
				self.spotprobs[i] = prob
				# self.spotprobs[i] = prob/d.max()

			self.gui.plot.image.set_array(self.disp_image)
			self.gui.docks['contrast'][1].change_contrast(-.05,1.05,0.0)
			self.gui.set_status('Finished')
			self.gui.app.processEvents()
			self.update_spots()

def calc_unimixmax_map(d,bg,nregion):
	from ..supporting import normal_minmax_dist as nmd

	## Most things should be background, but w/e.... equal a priori
	prior_bg = 0.5
	prior_uni = 0.5

	pbg = nmd.lnp_normal_max(d,nregion,bg[0],bg[1])
	pu = (d > bg[0])*-1.*np.log((d.max()-bg[0]))
	pp = 1./(1.+np.exp(pbg+np.log(prior_bg)-pu-np.log(prior_uni)))

	return pp

def calc_unimix_map(d,bg):
	from ..supporting import normal_minmax_dist as nmd

	## Most things should be background, but w/e.... equal a priori
	prior_bg = 0.5
	prior_uni = 0.5

	pbg = nmd.lnp_normal(d,bg[0],bg[1])
	pu = (d > bg[0])*-1.*np.log((d.max()-bg[0]))
	pp = 1./(1.+np.exp(pbg+np.log(prior_bg)-pu-np.log(prior_uni)))

	return pp


def run_todo(params):
	gmm, t,ti, i, mmax, h0, bg_values, nregion, nstates, maxiters, threshold, priors, update_bg = params

	out = gmm(h0, nstates, bg_values, nregion, initials=None, maxiters=maxiters, threshold=threshold, prior_strengths=priors[i],update_bg=update_bg)
	return [True,t,ti,mmax,1-out.r[:,0]]
