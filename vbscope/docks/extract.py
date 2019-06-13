from PyQt5.QtWidgets import QWidget, QSizePolicy,QLabel,QPushButton,QComboBox,QGridLayout,QFileDialog,QMessageBox
from PyQt5.QtCore import Qt

import numpy as np
import numba as nb
from scipy.ndimage import center_of_mass as com
from scipy.special import erf

import multiprocessing as mp
from time import time


default_prefs = {
	'extract_same_cutoff':1.42,
	'extract_numerical_aperture':1.2,
	'extract_pixel_size':13300.0,
	'extract_magnification':60.0,
	'extract_binning':2,
	'extract_nintegrate':7,
	'extract_ml_psf_maxiters':1000,
	'extract_fast_avg':False,
}

class dock_extract(QWidget):
	def __init__(self,parent=None):
		super(dock_extract, self).__init__(parent)

		self.default_prefs = default_prefs

		self.gui = parent

		self.button_extract = QPushButton('Extract')
		self.combo_method = QComboBox()
		self.combo_method.addItems(['Center','Average PSF','ML PSF'])
		self.combo_reduction = QComboBox()
		self.combo_reduction.addItems(['all','1','2','3','4'])

		self.button_plotter = QPushButton('Plots')
		self.button_save = QPushButton('Save')

		layout = QGridLayout()
		layout.addWidget(QLabel())
		layout.addWidget(QLabel("Colors:"),1,0)
		layout.addWidget(self.combo_reduction,1,1)
		layout.addWidget(QLabel("Method:"),2,0)
		layout.addWidget(self.combo_method,2,1)
		layout.addWidget(self.button_extract,3,0)
		layout.addWidget(self.button_save,4,0)
		layout.addWidget(self.button_plotter,4,1)
		self.setLayout(layout)

		self.combo_method.setCurrentIndex(1)
		self.combo_reduction.setCurrentIndex(0)

		self.button_extract.clicked.connect(self.extract)
		self.button_plotter.clicked.connect(self.launch_plotter)
		self.button_save.clicked.connect(self.save_traces)

		self.traces = None
		self.bgs=None


	def launch_plotter(self):
		self.gui.plot.clear_collections()
		self.gui.plot.canvas.draw()
		try:
			from smfret_plot import plotter_gui
			self.ui_p = plotter_gui(self.traces,gui=self.gui)
			self.ui_p.setWindowTitle('Plots')
			self.ui_p.show()
		except:
			msg = 'There was a problem trying to launch the smfret_plot program. Check python path'
			QMessageBox.critical(self,'Export Data',msg)


	def save_traces(self,event=None,oname=None):
		if self.traces is None:
			return
		if oname is None:
                    defaultname = self.gui.data.filename
                    if defaultname.count('.') > 0:
                        defaultname = '.'.join(defaultname.split('.')[:-1])
                    defaultname += '_traces.hdf5'
                    oname = QFileDialog.getSaveFileName(self.gui, 'Export data', defaultname,'*.hdf5')
		else:
			oname = [oname]
		if oname[0] != "":
			try:
				n = self.traces.shape[1]

				import h5py
				f = open(oname[0],'w')
				f.close()
				f = h5py.File(oname[0],'w')
				f.attrs['type'] = 'vbscope'
				f.attrs['ncolors'] = n
				f.create_dataset('data',data=self.traces,dtype='float32',compression="gzip")
				f.flush()
				f.create_dataset('pre_time',data=np.zeros(self.traces.shape[0],dtype='int32'),dtype='int32',compression="gzip")
				f.create_dataset('post_time',data=np.zeros(self.traces.shape[0],dtype='int32')+self.traces.shape[0],dtype='int32',compression="gzip")
				f.create_dataset('class',data=np.zeros(self.traces.shape[0],dtype='int8'),dtype='int8',compression="gzip")
				f.flush()
				f.close()
				self.gui.log("Exported data",True)

			except:
				msg = 'There was a problem trying to export the traces'
				QMessageBox.critical(self,'Export Data',msg)

	def get_spots(self):
		self.gui.statusbar.showMessage('Spot Finding')
		self.gui.app.processEvents()

		@nb.jit(["double[:,:](double[:,:],int64,int64)","int64[:,:](int64[:,:],int64,int64)"],nopython=True)
		def cull_rep_px(ss,nx,ny):
			x = ss[0]
			y = ss[1]

			m = np.zeros((nx,ny))
			for i in range(x.size):
				m[int(x[i]),int(y[i])] += 1

			x = []
			y = []
			for i in range(nx):
				for j in range(ny):
					if m[i,j] > 0:
						x.append(i)
						y.append(j)
			return np.array((x,y),dtype=ss.dtype)

		@nb.jit('double[:,:](double[:,:],double)',nopython=True)
		def avg_close(ss,cutoff):

			totalx = []
			totaly = []

			### AVERAGE
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
		    #
			# 		totalx.append(currentx/currentn)
			# 		totaly.append(currenty/currentn)
			# 		already.append(j)
			# return np.array((totalx,totaly))

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

		s = self.gui.docks['spotfind'][1].xys
		v = self.combo_reduction.currentIndex()

		regions,shifts = self.gui.data.regions_shifts()
		ts = self.gui.docks['transform'][1].transforms
		if ts is None:
			self.gui.statusbar.showMessage('No alignment. Get one')
			self.gui.app.processEvents()
			return

		self.gui.statusbar.showMessage('Spot Finding')
		self.gui.app.processEvents()


		for i in range(self.gui.data.ncolors):
			s[i] = cull_rep_px(s[i].astype('double'),self.gui.data.movie.shape[1],self.gui.data.movie.shape[2])

		if v == 0:
			if len(s[0]) >  1:
				ss = [s[i] - shifts[i][:,None] for i in range(self.gui.data.ncolors)]

				if self.gui.data.ncolors > 1:
					for j in range(1,self.gui.data.ncolors):
						# try:
							# cutoff = self.gui.prefs['same_cutoff']
							ss[j] = ts[j][0](ss[j].T).T
							# r = np.sqrt((ss[0][0,:,None]-ss[j][0,None,:])**2. + (ss[0][1,:,None]-ss[j][1,None,:])**2.)
							# rr = r.min(0)
							# cutoff = self.gui.prefs['same_cutoff']
							# cut = np.nonzero(np.sum((r < cutoff),axis=0) == 0)[0]
							# ss[j] = ss[j][:,cut]
						# except:
							# pass

			spots = np.concatenate(ss,axis=1)
			spots = avg_close(spots,self.gui.prefs['extract_same_cutoff'])

		else:
			i = v - 1
			spots = s[i] - shifts[i][:,None]
			if i != 0:
				spots = ts[0][i](spots.T).T
				spots = avg_close(spots,self.gui.prefs['extract_same_cutoff'])

		# print "Total spots: %d"%(spots.shape[1])
		return spots

	def get_sigma(self,j):
		''' j is the color index to pick the wavelenght of light '''

		c = 0.42 # .45 or .42, airy disk to gaussian
		psf_sig = c*self.gui.prefs['channels_wavelengths'][j]*self.gui.prefs['extract_numerical_aperture']
		sigma = psf_sig/self.gui.prefs['extract_pixel_size']*self.gui.prefs['extract_magnification']/self.gui.prefs['extract_binning']
		return sigma

	def extract(self):
		self.gui.statusbar.showMessage('Starting Extraction')

		if not self.gui.docks['transform'][1].flag_transforms:
			self.gui.statusbar.showMessage('Set alignment file')
			return
		spots = self.get_spots()

		regions,shifts = self.gui.data.regions_shifts()
		ts = self.gui.docks['transform'][1].transforms

		traces = []
		bgs = []

		for j in range(self.gui.data.ncolors):
			self.gui.statusbar.showMessage('Extracting Color %d'%(j))
			self.gui.app.processEvents()

			xy = spots.copy()
			if j > 0:
				xy = ts[j][0](xy.T).T
			xy += shifts[j][:,None]

			# xy = np.round(xy).astype('i')

			sigma = self.get_sigma(j)

			if self.combo_method.currentIndex() == 0:
				xyi = np.round(xy).astype('i')

				xyi[0][xyi[0] >= self.gui.data.movie.shape[1]] = self.gui.data.movie.shape[1] - 1

				xyi[1][xyi[1] >= self.gui.data.movie.shape[2]] = self.gui.data.movie.shape[2] - 1
				ns = self.gui.data.movie[:,xyi[0],xyi[1]]

				# ## get global background
				# from supporting import minmax
				# from supporting import normal_minmax_dist as nd
				# r = regions[j]
				# imm = self.gui.data.movie[:,r[0][0]:r[0][1],r[1][0]:r[1][1]].min(0)
				# mmin,mmax = minmax.minmax_map(imm,self.gui.prefs['nsearch'],self.gui.prefs['clip border'])
				# bgfit = nd.estimate_from_min(imm[mmin],self.gui.prefs['nsearch']**2 * imm.shape[0])

				# ## Good - bg is median of four corners for everyframe for every spot. dynamic
				# bs = np.median(np.array([self.gui.data.movie[:,xyi[0]+ii,xyi[1]+jj] for ii,jj in zip([-2,-2,2,2],[-2,2,-2,2])] ),axis=0)

				## Okay - bg is mean infered from min-value order statistics of four corners for every spot. static
				## This approach errors out if xyi + 2 > movie size...
				# from ..supporting import normal_minmax_dist as nd
				# bs = np.array([self.gui.data.movie[:,xyi[0]+ii,xyi[1]+jj] for ii,jj in zip([-2,-2,2,2],[-2,2,-2,2])] )
				# bs = np.min(bs,axis=0)
				# bgg = np.zeros(bs.shape[1])
				# for i in range(bgg.size):
				# 	bgg[i] = nd.estimate_from_min(bs[:,i], 4.)[0]
				# bs = bgg[None,:]



				# bgg[i] = np.median(self.gui.data.movie[-10,xyi[0][i],xyi[1][i]])
				# from ..supporting import normal_minmax_dist as nd
				# global_bg = np.percentile(np.median(self.gui.data.movie[-10:],axis=0).flatten(),1.)
				bgg = np.zeros(ns.shape[1])# + global_bg
				for i in range(bgg.size):
					try:
						bs = np.array([self.gui.data.movie[-10:,xyi[0][i]+ii,xyi[1][i]+jj] for ii,jj in zip([-2,-2,2,2],[-2,2,-2,2])])

						bs = np.min(bs,axis=0)
						bgg[i] = nd.median(bs)
						# bgg[i] = nd.estimate_from_min(bs, 4.)[0]
					except:
						bgg[i] = np.median(self.gui.data.movie[-10:,xyi[0][i],xyi[1][i]])
				ns = ns - bgg[None,:]


				traces.append(ns)

			elif self.combo_method.currentIndex() == 1:
				# ns = self.ml_psf(np.round(xy).astype('i'),sigma)
				# traces.append(ns)
				ns = self.ml_psf(xy,sigma,j)
				traces.append(ns)

			elif self.combo_method.currentIndex() == 2:
				ns = self.experimental(np.round(xy).astype('i'),sigma)
				traces.append(ns)

		traces = np.array(traces)
		traces = np.moveaxis(traces,2,0)

		# np.save('test.dat',traces)
		self.traces = traces
		self.gui.statusbar.showMessage('Traces Extracted')


	def cancel_expt(self):
		self.flag_cancel = True

	def experimental(self,xy,sigma):
		l = (self.gui.prefs['extract_nintegrate']-1)/2
		out = np.empty((self.gui.data.movie.shape[0],xy.shape[1]))
		prog = progress(out.shape[0],out.shape[1])
		prog.canceled.connect(self.cancel_expt)
		prog.show()
		self.gui.app.processEvents()
		self.flag_cancel = False

		self.gui.set_status('Compiling...')
		from ..supporting.ml_fit import fit
		self.gui.set_status('')
		for t in range(self.gui.data.movie.shape[0]):
			z = self.gui.data.movie[t].astype('double')
			if not self.flag_cancel:
				prog.setValue(t)
				self.gui.app.processEvents()
				t0 = time()
				if self.gui.prefs['computer_ncpu'] > 1:
					pool = mp.Pool(self.gui.prefs['computer_ncpu'])
					ps = pool.map(_fit_wrapper,[[l,z,sigma,xy[:,i].astype('double')] for i in range(xy.shape[1])])
					pool.close()
				else:
					ps = list(map(_fit_wrapper,[(l,z,sigma,xy[:,i].astype('double')) for i in range(xy.shape[1])]))
				for i in range(xy.shape[1]):
					# xyi = xy[:,i].astype('double')
					# p = fit(l,z,sigma,xyi)
					# print t,i,p[4],p[5]
					# out[t,i] = p[4]
					out[t,i] = ps[i][4]
				t1 = time()
				prog.setLabelText('Fitting %d spots, %d frames\ntime/fit = %f sec'%(out.shape[0],out.shape[1],(t1-t0)/out.shape[1]))
		prog.close()
		return out

	def ml_psf(self,xy,sigma,color):
		self.gui.set_status('Compiling...')
		from ..supporting.ml_fit import ml_psf
		self.gui.set_status('')
		from time import time

		l = (self.gui.prefs['extract_nintegrate']-1)/2
		out = np.zeros((self.gui.data.movie.shape[0],xy.shape[1]))

		prog = progress(out.shape[0],out.shape[1])
		prog.setWindowTitle("Fitting Color %d"%(color))
		prog.setRange(0,out.shape[1])
		prog.canceled.connect(self.cancel_expt)
		prog.show()
		self.gui.app.processEvents()
		self.flag_cancel = False

		ts = [0.0]
		for i in range(out.shape[1]):
			if not self.flag_cancel:
				if i%10 == 0:
					if i > 0:
						prog.setLabelText('Fitting spot %d/%d\nAvg. time/fit = %f s'%(i+1,out.shape[1],np.mean(ts[-10:])))
					prog.setValue(i)
					self.gui.app.processEvents()
				try:
					t0 = time()
					o = ml_psf(l,self.gui.data.movie,sigma,xy[:,i].astype('double'),maxiters=self.gui.prefs['extract_ml_psf_maxiters'],fastflag=self.gui.prefs['extract_fast_avg'])
					t1 = time()
					out[:,i] = o
					ts.append(t1-t0)
				except:
					pass
		prog.close()
		return out

def _fit_wrapper(params):
	from ..supporting.ml_fit import fit
	return fit(*params)

from PyQt5.QtWidgets import QProgressDialog
class progress(QProgressDialog):
	def __init__(self,nmax,tmax):
		QProgressDialog.__init__(self)
		from PyQt5.Qt import QFont
		self.setFont(QFont('monospace'))
		self.setWindowTitle("Fitting Spots")
		self.setLabelText('Fitting %d spots, %d frames\ntime/fit = 0.0 sec'%(nmax,tmax))
		self.setRange(0,tmax)
