from PyQt5.QtWidgets import QWidget, QSizePolicy, QPushButton,QGridLayout,QComboBox,QLineEdit,QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator

import numpy as np

from scipy.ndimage import median_filter,gaussian_filter,minimum_filter,uniform_filter

default_prefs = {
	'background_pixel_dist':3,
	'background_time_dist':10,
	'background_smooth_dist':3
}

# from ..supporting import gpu_flatfield as gf

class dock_background(QWidget):
	def __init__(self,parent=None):
		super(dock_background, self).__init__(parent)

		self.default_prefs = default_prefs

		self.gui = parent

		self.flag_showing = False

		self.radius1 = 10.
		self.radius2 = 5.

		layout = QGridLayout()

		self.combo_method = QComboBox()
		self.button_preview = QPushButton('Toggle Preview')
		self.le_radius1 = QLineEdit()
		self.le_radius2 = QLineEdit()

		self.le_radius1.setValidator(QDoubleValidator(1e-300,1e300,100))
		self.le_radius2.setValidator(QDoubleValidator(1e-300,1e300,100))
		self.le_radius1.editingFinished.connect(self.update_radius1)
		self.le_radius2.editingFinished.connect(self.update_radius2)
		self.le_radius1.setText(str(self.radius1))
		self.le_radius2.setText(str(self.radius2))

		layout.addWidget(QLabel('Method:'),0,0)
		layout.addWidget(self.combo_method,0,1)
		layout.addWidget(QLabel('Filter Radius (px):'),1,0)
		layout.addWidget(self.le_radius1,1,1)
		layout.addWidget(QLabel('Smoothing Radius (px):'),2,0)
		layout.addWidget(self.le_radius2,2,1)
		layout.addWidget(self.button_preview,3,1)

		# self.button_flat = QPushButton("Flat Field")
		# self.button_pseudo = QPushButton("Pseudo Flat Field")
		# self.button_dynamic = QPushButton("Dynamic Imaging")
		# self.button_differential = QPushButton("Differential Imaging")
		# self.button_ratiometric = QPushButton("Ratiometric Imaging")
		# self.button_normalize = QPushButton("Normalize Power")
		# self.button_bin = QPushButton("Bin 2x")
		#
		# layout.addWidget(self.button_flat,4,0)
		# layout.addWidget(self.button_pseudo,4,1)
		# layout.addWidget(self.button_normalize,4,2)
		# layout.addWidget(self.button_bin,4,3)
		# layout.addWidget(self.button_differential,5,0)
		# layout.addWidget(self.button_dynamic,5,1)
		# layout.addWidget(self.button_ratiometric,5,2)
		#
		# self.button_flat.clicked.connect(self.flat)
		# self.button_pseudo.clicked.connect(self.pseudo)
		# self.button_normalize.clicked.connect(self.normalize)
		# self.button_bin.clicked.connect(self.bin)
		# self.button_differential.clicked.connect(self.differential)
		# self.button_dynamic.clicked.connect(self.dynamic)
		# self.button_ratiometric.clicked.connect(self.ratiometric)

		self.setLayout(layout)

		self.button_preview.clicked.connect(self.preview)

		self.combo_method.addItems(['None','Minimum','Median','Uniform','Test'])
		self.method = 1
		self.combo_method.setCurrentIndex(self.method)
		self.combo_method.currentIndexChanged.connect(self.update_method)

	def prep(self):
		nc = self.gui.data.ncolors
		regions,shifts = self.gui.data.regions_shifts()
		prefs = self.gui.prefs
		self.gui.data.movie = self.gui.data.movie.astype('float32')
		self.gui.plot.image.set_array(self.gui.plot.image.get_array().astype('float32'))
		return nc,regions,shifts,prefs
	# 
	# def flat(self):
	# 	nc,regions,shifts,p = self.prep()
	#
	# 	for i in range(nc):
	# 		r = regions[i]
	# 		mtype = self.gui.data.movie.dtype
	# 		d = self.gui.data.movie[:,r[0][0]:r[0][1],r[1][0]:r[1][1]].astype('f').copy()
	# 		self.gui.data.movie[:,r[0][0]:r[0][1],r[1][0]:r[1][1]] = gf.flat_field(d,d[:-1])#.astype(mtype)
	#
	#
	# def pseudo(self):
	# 	nc,regions,shifts,p = self.prep()
	#
	# 	for i in range(nc):
	# 		r = regions[i]
	# 		mtype = self.gui.data.movie.dtype
	# 		d = self.gui.data.movie[:,r[0][0]:r[0][1],r[1][0]:r[1][1]].astype('f').copy()
	# 		self.gui.data.movie[:,r[0][0]:r[0][1],r[1][0]:r[1][1]] = gf.pseudo_flat_field(d,11)#.astype(mtype)
	#
	#
	# def normalize(self):
	# 	nc,regions,shifts,p = self.prep()
	#
	# 	for i in range(nc):
	# 		r = regions[i]
	# 		mtype = self.gui.data.movie.dtype
	# 		d = self.gui.data.movie[:,r[0][0]:r[0][1],r[1][0]:r[1][1]].astype('f').copy()
	# 		self.gui.data.movie[:,r[0][0]:r[0][1],r[1][0]:r[1][1]] = gf.normalize_movie_power(d)#.astype(mtype)
	#
	# def bin(self):
	# 	self.gui.data.movie = gf.bin2x(self.gui.data.movie)
	#
	# def differential(self):
	# 	nc,regions,shifts,p = self.prep()
	#
	# 	for i in range(nc):
	# 		r = regions[i]
	# 		mtype = self.gui.data.movie.dtype
	# 		n = 99
	# 		d = self.gui.data.movie[:,r[0][0]:r[0][1],r[1][0]:r[1][1]].astype('f').copy()
	# 		self.gui.data.movie[:,r[0][0]:r[0][1],r[1][0]:r[1][1]] = gf.differential_imaging(d,n)#.astype(mtype)
	# 		self.gui.data.movie[:,r[0][0]:r[0][1],r[1][0]:r[1][1]] = 0
	#
	# def dynamic(self):
	# 	nc,regions,shifts,p = self.prep()
	#
	# 	for i in range(nc):
	# 		r = regions[i]
	# 		mtype = self.gui.data.movie.dtype
	# 		d = self.gui.data.movie[:,r[0][0]:r[0][1],r[1][0]:r[1][1]].astype('f').copy()
	# 		self.gui.data.movie[:,r[0][0]:r[0][1],r[1][0]:r[1][1]] = gf.dynamic_imaging(d)#.astype(mtype)
	#
	# def ratiometric(self):
	# 	nc,regions,shifts,p = self.prep()
	#
	# 	for i in range(nc):
	# 		r = regions[i]
	# 		mtype = self.gui.data.movie.dtype
	# 		d = self.gui.data.movie[:,r[0][0]:r[0][1],r[1][0]:r[1][1]].astype('f').copy()
	# 		self.gui.data.movie[:,r[0][0]:r[0][1],r[1][0]:r[1][1]] = gf.ratiometric_imaging(d,20)#.astype(mtype)


	def update_radius1(self):
		self.radius1 = float(self.le_radius1.text())
		self.update_background()

	def update_radius2(self):
		self.radius2 = float(self.le_radius2.text())
		self.update_background()

	def update_method(self,i):
		self.method = i
		self.update_background()
		self.draw_background()

	def calc_background(self,image):
		if self.method == 1:#'Minimum':
			# return gaussian_filter(minimum_filter(image,self.radius1),self.radius2)
			return minimum_filter(gaussian_filter(image,self.radius2),int(self.radius1))
		elif self.method == 2:#'Median':
			# return gaussian_filter(median_filter(image,int(self.radius1)),self.radius2)
			return median_filter(gaussian_filter(image,self.radius2),int(self.radius1))
		elif self.method == 3:#new
			return uniform_filter(gaussian_filter(image,self.radius2),int(self.radius1))
		elif self.method == 4:
			x =  self.test()
			return x#image*(x-1.)/x
		else:
			return np.zeros(image.shape,dtype='f')

	def update_background(self):
		if self.gui.data.flag_movie:
			self.gui.data.background = self.calc_background(self.gui.data.movie[self.gui.data.current_frame].astype('f'))

	def draw_background(self):
		if self.gui.data.flag_movie:
			if self.flag_showing:
				self.gui.plot.image.set_array(self.gui.data.background)
			else:
				self.gui.plot.image.set_array(self.gui.data.movie[self.gui.data.current_frame] - self.gui.data.background)
			self.gui.docks['contrast'][1].update_image_contrast()
			self.gui.plot.draw()

	def preview(self):
		self.flag_showing = ~self.flag_showing
		self.update_background()
		self.draw_background()

	def test(self):
		# from PyQt5.QtWidgets import QMessageBox
		# power = np.median(self.gui.data.movie,axis=(1,2))
		# power /= np.median(power)
		#
		# from ..supporting import solve_bg
		# background = solve_bg(self.gui.data.movie / power[:,None,None], m=self.gui.prefs['background_pixel_dist'], n=self.gui.prefs['background_time_dist'], sigma=self.gui.prefs['background_smooth_dist']).astype('f')
		# # background /= np.median(background)
		# return background


		m = self.gui.data.movie.mean(0)
		a = median_filter(m,24)
		b = median_filter(a,24)
		c = median_filter(b,24)
		return c
