from PyQt5.QtWidgets import QWidget, QSizePolicy,QLabel,QPushButton,QComboBox,QGridLayout
from PyQt5.QtCore import Qt

import numpy as np
from scipy.ndimage import center_of_mass as com
from scipy.special import erf

class dock_extract(QWidget):
	def __init__(self,parent=None):
		super(dock_extract, self).__init__(parent)

		self.gui = parent

		self.button_extract = QPushButton('Extract')
		self.combo_method = QComboBox()
		self.combo_method.addItems(['Center','Radial Sum','Average PSF','ML PSF'])
		self.combo_reduction = QComboBox()
		self.combo_reduction.addItems(['all','1','2','3','4'])

		self.button_plotter = QPushButton('Plots')

		layout = QGridLayout()
		layout.addWidget(QLabel("Colors:"),0,0)
		layout.addWidget(self.combo_reduction,0,1)
		layout.addWidget(QLabel("Method:"),1,0)
		layout.addWidget(self.combo_method,1,1)
		layout.addWidget(self.button_extract,2,0)
		layout.addWidget(self.button_plotter,2,1)
		self.setLayout(layout)

		self.combo_method.setCurrentIndex(2)
		self.combo_reduction.setCurrentIndex(0)

		self.button_extract.clicked.connect(self.extract)
		self.button_plotter.clicked.connect(self.launch_plotter)

		self.traces = None
		self.bgs=None


	def launch_plotter(self):
		self.gui.plot.clear_collections()
		self.gui.plot.canvas.draw()
		from plotter import ui_plotter
		self.ui_p = ui_plotter(self.traces,self.get_spots(),self.bgs,self)
		self.ui_p.setWindowTitle('Plots')
		self.ui_p.show()

	def get_spots(self):
		self.gui.statusbar.showMessage('Determining which spots')
		self.gui.app.processEvents()

		s = self.gui.docks['spotfind'][1].xys
		v = self.combo_reduction.currentIndex()

		regions,shifts = self.gui.data.regions_shifts()
		ts = self.gui.docks['transform'][1].transforms

		if v == 0:
			if len(s[0]) >  1:
				ss = [s[i] - shifts[i][:,None] for i in range(self.gui.data.ncolors)]
				if self.gui.data.ncolors > 1:
					for j in range(1,self.gui.data.ncolors):
						try:
							ss[j] = ts[j][0](ss[j].T).T
							r = np.sqrt((ss[0][0,:,None]-ss[j][0,None,:])**2. + (ss[0][1,:,None]-ss[j][1,None,:])**2.)
							rr = r.min(0)
							cutoff = self.gui.prefs['same_cutoff']
							cut = np.nonzero(np.sum((r < cutoff),axis=0) == 0)[0]
							ss[j] = ss[j][:,cut]
						except:
							pass
			spots = np.concatenate(ss,axis=1)
		else:
			i = v - 1
			spots = s[i] - shifts[i][:,None]
			if i != 0:
				spots = ts[0][i](spots.T).T
		return spots

	def get_sigma(self,j):
		''' j is the color index to pick the wavelenght of light '''

		c = 0.42 # .45 or .42, airy disk to gaussian
		psf_sig = c*self.gui.prefs['wavelengths_nm'][j]*self.gui.prefs['numerical_aperture']
		sigma = psf_sig/self.gui.prefs['pixel_size']*self.gui.prefs['magnification']/self.gui.prefs['binning']
		return sigma

	def extract(self):
		self.gui.statusbar.showMessage('Starting Extraction')

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
				traces.append(self.gui.data.movie[:,xyi[0],xyi[1]])

			elif self.combo_method.currentIndex() == 1:
				l = (self.gui.prefs['nintegrate']-1)/2
				ns = []
				bs = []
				for i in xrange(xy.shape[1]):
					xyi = np.round(xy[:,i]).astype('i')
					xmin = np.max((0,xyi[0]-l))
					xmax = np.min((self.gui.data.movie.shape[1]-1,xyi[0]+l))
					ymin = np.max((0,xyi[1]-l))
					ymax = np.min((self.gui.data.movie.shape[2]-1,xyi[1]+l))

					gx,gy = np.mgrid[xmin:xmax+1,ymin:ymax+1]
					m = self.gui.data.movie[:,xmin:xmax+1,ymin:ymax+1].astype('f')
					xyi = com(m.max(0)) + xy[:,i] - l

					dex = .5 * (erf((xy[0,i]-gx+.5)/(np.sqrt(2.*sigma**2.)))
						- erf((xy[0,i]-gx -.5)/(np.sqrt(2.*sigma**2.))))
					dey = .5 * (erf((xy[1,i]-gy+.5)/(np.sqrt(2.*sigma**2.)))
						- erf((xy[1,i]-gy -.5)/(np.sqrt(2.*sigma**2.))))
					psi = dex*dey
				#
				# 	b = np.min(m*(1.-psi[None,:,:]),axis=(1,2))
				# 	# b = np.mean(m*(1.-psi[None,:,:]),axis=(1,2))
				# 	n = ((m-b[:,None,None])*psi[None,:,:]).sum((1,2))/np.sum(psi**2.)

					# dr = np.sqrt((gx.astype('f')-xy[0])**2. + (gy.astype('f')-xy[1])**2.)
					# psi = 1./np.sqrt(2.*np.pi*sigma**2.) * np.exp(-.5/sigma/sigma*(dr-0.)**2.)
					psi /= psi.sum()

					b = np.mean(m*(1.-psi[None,:,:]),axis=(1,2))
					n = ((m-b[:,None,None])*psi[None,:,:]).sum((1,2))
					bs.append(b)
					ns.append(n)

				ns = np.array(ns).T
				traces.append(ns)
				bs = np.array(bs).T
				bgs.append(bs)

			elif self.combo_method.currentIndex() == 2:
				l = (self.gui.prefs['nintegrate']-1)/2
				ns = []
				bs = []
				for i in xrange(xy.shape[1]):
					xyi = np.round(xy[:,i]).astype('i')
					xmin = np.max((0,xyi[0]-l))
					xmax = np.min((self.gui.data.movie.shape[1]-1,xyi[0]+l))
					ymin = np.max((0,xyi[1]-l))
					ymax = np.min((self.gui.data.movie.shape[2]-1,xyi[1]+l))

					gx,gy = np.mgrid[xmin:xmax+1,ymin:ymax+1]
					gx = gx.astype('f')
					gy = gy.astype('f')
					m = self.gui.data.movie[:,xmin:xmax+1,ymin:ymax+1].astype('f')
					xyi = com(m.max(0)) + xy[:,i] - l

					dex = .5 * (erf((xy[0,i]-gx+.5)/(np.sqrt(2.*sigma**2.)))
						- erf((xy[0,i]-gx -.5)/(np.sqrt(2.*sigma**2.))))
					dey = .5 * (erf((xy[1,i]-gy+.5)/(np.sqrt(2.*sigma**2.)))
						- erf((xy[1,i]-gy -.5)/(np.sqrt(2.*sigma**2.))))
					psi = dex*dey

					# b = np.mean(m*(1.-psi[None,:,:]),axis=(1,2))
					b = np.mean(m,axis=(1,2))
					n = self.gui.data.movie[:,np.round(xyi[0]).astype('i'),np.round(xyi[1]).astype('i')]
					# n = ((m-b[:,None,None])*psi[None,:,:]).sum((1,2))/np.sum(psi**2.)

					n0 = n.sum()
					psum = np.sum(psi**2.)
					for it in xrange(100):
						b = np.mean(m - n[:,None,None]*psi[None,:,:],axis=(1,2))
						# b = b.mean()
						# n = np.sum((m - b)*psi[None,:,:] ,axis=(1,2)) / psum
						n = np.sum((m - b[:,None,None])*psi[None,:,:] ,axis=(1,2))/np.sum(psi**2.)
						n1 = n.sum()
						if np.isclose(n1,n0):
							break
						else:
							n0 = n1
					ns.append(n)
					bs.append(b)
				ns = np.array(ns).T
				traces.append(ns)
				bs = np.array(bs).T
				bgs.append(bs)

		traces = np.array(traces)
		traces = np.moveaxis(traces,2,0)
		# np.save('test.dat',traces)
		self.traces = traces
		try:
			bgs = np.array(bgs)
			bgs = np.moveaxis(bgs,2,0)
			self.bgs = bgs
			# np.save('test_bg.dat',bgs)
		except:
			self.bgs = None

		self.gui.statusbar.showMessage('Traces Extracted')
