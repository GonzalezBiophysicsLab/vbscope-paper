from PyQt5.QtWidgets import QWidget, QSizePolicy,QGridLayout,QLabel,QSpinBox,QMessageBox, QFileDialog, QPushButton
from PyQt5.QtCore import Qt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

import numpy as np


default_prefs = {
	'transform_alignment_order':4
}

class dock_transform(QWidget):
	def __init__(self,parent=None):
		super(dock_transform, self).__init__(parent)

		self.default_prefs = default_prefs

		self.gui = parent

		self.flag_sectors = False
		self.flag_transforms = False
		self.flag_spots = False

		self.layout = QGridLayout()

		l = QLabel(r'N colors')
		self.spin_colors = QSpinBox()
		self.spin_colors.setRange(1,4)
		self.spin_colors.setValue(self.gui.data.ncolors)

		self.button_preview = QPushButton('Toggle Sectors')
		self.button_toggle = QPushButton('Toggle Aligned')
		self.button_load = QPushButton('Load Alignment File')
		self.button_estimate = QPushButton('Estimate from Spots')
		self.button_fftestimate = QPushButton('Estimate from Image')
		self.button_export = QPushButton('Export Alignment')

		self.label_load = QLabel('')

		self.layout.addWidget(l,1,0)
		self.layout.addWidget(self.spin_colors,1,1)

		self.layout.addWidget(self.button_load,2,0)
		self.layout.addWidget(QLabel(),0,0)
		self.layout.addWidget(self.button_estimate,3,0)
		self.layout.addWidget(self.button_fftestimate,3,1)
		self.layout.addWidget(self.button_export,2,1)
		self.layout.addWidget(self.button_preview,4,0)
		self.layout.addWidget(self.button_toggle,4,1)

		self.button_export.clicked.connect(self.export)
		self.button_preview.clicked.connect(self.preview)
		self.button_toggle.clicked.connect(self.toggle)
		self.button_load.clicked.connect(self.load)
		self.button_estimate.clicked.connect(self.stochastic)
		self.button_fftestimate.clicked.connect(self.fft_estimate)

		self.spin_colors.valueChanged.connect(self.update_colors)

		self.setLayout(self.layout)
		self.update_colors(self.gui.data.ncolors)

	def export(self):
		if self.flag_transforms:
			n = self.gui.data.ncolors
			regions,shifts = self.gui.data.regions_shifts()
			if n > 1:

				# m = 1000
				# gx = np.random.uniform(regions[0][0][0],regions[0][0][1],size=m)
				# gy = np.random.uniform(regions[0][1][0],regions[0][1][1],size=m)
				# g = np.array((gx,gy))
				# out = [g[1],g[0]]
				# for j in range(1,n):
				# 	o = self.transforms[0][j](g.T).T
				# 	out.append(o[1])
				# 	out.append(o[0])
				# out = np.array((out)).T
				m = 5
				gx,gy = np.mgrid[regions[0][0][0]:regions[0][0][1]:m,regions[0][1][0]:regions[0][1][1]:m]
				g = np.array((gx.flatten(),gy.flatten()))
				out = [g[1],g[0]]
				for j in range(1,n):
					o = self.transforms[j][0](g.T).T
					out.append(o[1])
					out.append(o[0])
				out = np.array((out)).T + 1.

				oname = QFileDialog.getSaveFileName(self, 'Export vbscope Alignment', self.gui.data.filename[:-4]+'_alignment.dat','*.dat')
				if oname[0] != "":
					try:
						np.savetxt(oname[0],out,delimiter=',')
					except:
						QMessageBox.critical(self,'Export Status','There was a problem trying to export the alignment locations')

	def toggle(self):
		if self.flag_transforms and self.gui.docks['spotfind'][1].flag_spots:
			self.flag_spots = not self.flag_spots
			if self.flag_spots:
				self.plot_overlapped()
			else:
				self.gui.plot.clear_collections()
			self.gui.plot.canvas.draw()

	def preview(self):
		if self.gui.data.flag_movie:
			self.flag_sectors = not self.flag_sectors
			if self.flag_sectors:
				self.update_colors(self.gui.data.ncolors)
			else:
				self.gui.plot.clear_collections()
				# self.gui.plot.ax.collections[0].remove()
			self.gui.plot.canvas.draw()

	def safeload(self,fname):
		f = open(fname,'r')
		l = f.readline()
		f.close()
		if l.count(',') > 0:
			delim = ','
		else:
			delim = ' '
		return np.loadtxt(fname,delimiter=delim)

	def load(self,event=None,fname=None):
		if fname is None:
			fname = QFileDialog.getOpenFileName(self,'Choose an alignment file to load','./')#,filter='TIF File (*.tif *.TIF)')
		else:
			fname = [fname]
		if fname[0] != "":
			try:
				d = self.safeload(fname[0])
				success = True
			except:
				success = False

			if success:
				from ..supporting import transforms
				self.label_load.setText('%d x %d'%(d.shape[0],d.shape[1]))
				self.gui.statusbar.showMessage('Loaded Alignment: %s'%(fname[0]))

				#### ASSSUME VBSCOPE (MATLAB) FORMAT
				ncolor = self.gui.data.ncolors
				dd = [d[:,2*i:2*i+2][:,::-1].T - 1. for i in range(d.shape[1]//2)]
				self.estimate(dd)
				self.plot_overlapped(dd)

			else:
				QMessageBox.critical(None,'Could Not Load File','Could not load file: %s.'%(fname[0]))

	def update_colors(self,v):
		self.gui.data.ncolors = v
		# if self.gui.data.ncolors == 2:
		# 	self.gui.plot.colorlist = ['lime','red']
		# else:
		# 	self.gui.plot.colorlist = self.gui.plot.colorlist_ordered

		# self.gui.docks['spotfind'][1].show_spins()

		if self.gui.data.flag_movie:
			c = self.gui.prefs['channels_colors']#self.gui.plot.colorlist
			alpha = .4

			self.flag_sectors = True
			self.gui.plot.clear_collections()

			regions,shifts = self.gui.data.regions_shifts()

			rs = []
			for i in range(self.gui.data.ncolors):
				r = Rectangle((shifts[i][1], shifts[i][0]), regions[i][1][1] - regions[i][1][0], regions[i][0][1] - regions[i][0][0], ec=c[i], fc=c[i], fill=True, alpha=alpha, lw=3)
				rs.append(r)
			pc = PatchCollection(rs,match_original=True)

			self.gui.plot.ax.add_collection(pc)
			self.gui.plot.canvas.draw()

	def stochastic(self):
		if self.gui.docks['spotfind'][1].flag_spots:
			self.estimate()
			self.plot_overlapped()

	def get_images(self):
		sfd = self.gui.docks['spotfind'][1]

		nc = self.gui.data.ncolors
		start = sfd.spin_start.value() - 1
		end = sfd.spin_end.value() - 1
		# d = np.mean(self.gui.data.movie[start:end+1],axis=0)

		regions,self.shifts = self.gui.data.regions_shifts()

		imgs = []
		for i in range(nc):
			r = regions[i]
			clip = self.gui.prefs['spotfind_clip_border']
			d = self.gui.data.movie[start:end+1,r[0][0]+clip:r[0][1]-clip,r[1][0]+clip:r[1][1]-clip].astype('f').mean(0)
			# bg = self.gui.docks['background'][1].calc_background(d).astype('f')
			# dd = d.copy() - bg
			dd = self.gui.docks['background'][1].bg_filter(d)

			# r = regions[i]
			# dd = d[r[0][0]:r[0][1],r[1][0]:r[1][1]].astype('f').copy()
			# dd -= self.gui.docks['background'][1].calc_background(dd).astype('f')
			imgs.append(dd)
		return imgs

	def fft_estimate(self):
		if self.gui.docks['spotfind'][1].flag_spots:
			from ..supporting import transforms

			imgs = self.get_images()
			nc = self.gui.data.ncolors

			ts = []
			for i in range(nc):
				tts = [None for j in range(nc)]
				for j in range(nc):
					if i != j:
						s1,s2,_,tform = transforms.interpolated_fft_phase_alignment(imgs[j],imgs[i])
						tts[j] = tform
				ts.append(tts)
			self.transforms = ts
			self.flag_transforms = True
			self.gui.statusbar.showMessage('Finished Finding Transforms')
			self.plot_overlapped()


	def estimate(self,cs=None):
		from ..supporting import transforms

		regions,shifts = self.gui.data.regions_shifts()

		imgs = self.get_images()

		ts = []
		for i in range(self.gui.data.ncolors):
			tts = [None for j in range(self.gui.data.ncolors)]
			for j in range(self.gui.data.ncolors):
				if i != j:
					if cs is None:
						c1 = self.gui.docks['spotfind'][1].xys[i].copy()
						c2 = self.gui.docks['spotfind'][1].xys[j].copy()

						for ii in range(2):
							c1[ii] -= shifts[i][ii]
							c2[ii] -= shifts[j][ii]

						# tts[j] = transforms.icp(c1.T.astype('f'),c2.T.astype('f'),1e-6,1e-6,maxiters=100)

						s1,s2,_,tform = transforms.interpolated_fft_phase_alignment(imgs[j],imgs[i])

						#### QUITE POSSIBLE THAT IT SHOULD BE S2,S1... or -S1,-S2... or -S2,-S1.... INSTEAD OF S1,S2....
						tts[j] = transforms.icp(c1.T.astype('f'),c2.T.astype('f'),s1,s2,maxiters=100)
						# tts[j] = transforms.icp(c1.T.astype('f'),c2.T.astype('f'),0.,0.,maxiters=100)
						self.gui.log("Alignment Loaded - ICP")

					else:
						c1 = cs[i].copy()
						c2 = cs[j].copy()
						tts[j] = transforms.poly(c1.T.astype('f'),c2.T.astype('f'),order=self.gui.prefs['transform_alignment_order'])
						# from skimage.transform import AffineTransform,EuclideanTransform,PolynomialTransform
						# tts[j] = EuclideanTransform(translation=np.median(c1,axis=1) - np.median(c2,axis=1))
						# tts[j].estimate(c1.T.astype('f'),c2.T.astype('f'))
						if j == 0 and i == 1:
							self.gui.log("Alignment Loaded - order=%d polynomial, residual: %0.3f"%(self.gui.prefs['transform_alignment_order'],np.median(tts[0].residuals(c2.T.astype('f'),c1.T.astype('f')))))

			ts.append(tts)


		self.transforms = ts
		self.flag_transforms = True
		self.gui.statusbar.showMessage('Finished Finding Transforms')

	def plot_overlapped(self,cs = None):
		colors = self.gui.prefs['channels_colors']#self.gui.plot.colorlist
		self.gui.plot.clear_collections()
		regions,shifts = self.gui.data.regions_shifts()

		for i in range(self.gui.data.ncolors):
			for j in range(self.gui.data.ncolors):
				if i != j:
					if cs is None:
						c1 = self.gui.docks['spotfind'][1].xys[i].copy()
						c2 = self.gui.docks['spotfind'][1].xys[j].copy()
						for ii in range(2):
							c1[ii] -= shifts[i][ii]
							c2[ii] -= shifts[j][ii]
					else:
						c1 = cs[i].copy()
						c2 = cs[j].copy()

					ct2 = self.transforms[i][j](c2.T).T

					for ii in range(2):
						c1[ii] += shifts[i][ii]
						ct2[ii] += shifts[i][ii]
					self.gui.plot.scatter(c1[0],c1[1],color=colors[i])
					self.gui.plot.scatter(ct2[0],ct2[1],color=colors[j])
					self.flag_transforms = True
		self.gui.plot.canvas.draw()
