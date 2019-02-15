from PyQt5.QtWidgets import QWidget, QSizePolicy, QPushButton,QGridLayout,QComboBox,QLineEdit,QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator

import numpy as np

# from scipy.ndimage import median_filter,gaussian_filter,minimum_filter,uniform_filter

default_prefs = {
	# 'background_pixel_dist':3,
	# 'background_time_dist':10,
	# 'background_smooth_dist':3
}

# from ..supporting import gpu_flatfield as gf

class dock_background(QWidget):
	def __init__(self,parent=None):
		super(dock_background, self).__init__(parent)

		self.default_prefs = default_prefs
		self.gui = parent

		self.method = 2
		self.radius1 = .5
		self.radius2 = 10

		layout = QGridLayout()

		self.combo_method = QComboBox()
		self.le_radius1 = QLineEdit()
		self.le_radius2 = QLineEdit()

		self.le_radius1.setValidator(QDoubleValidator(1e-300,1e300,100))
		self.le_radius2.setValidator(QDoubleValidator(1e-300,1e300,100))
		self.le_radius1.editingFinished.connect(self.update_radius1)
		self.le_radius2.editingFinished.connect(self.update_radius2)
		self.le_radius1.setText(str(self.radius1))
		self.le_radius2.setText(str(self.radius2))

		self.button_removebg = QPushButton('Remove BG')
		self.button_removebg.clicked.connect(self.remove_bg)

		layout.addWidget(QLabel(),0,0)
		layout.addWidget(QLabel('Method:'),1,0)
		layout.addWidget(self.combo_method,1,1)
		layout.addWidget(QLabel('Filter Radius (close)'),2,0)
		layout.addWidget(self.le_radius1,2,1)
		layout.addWidget(QLabel('Filter Radius (far):'),3,0)
		layout.addWidget(self.le_radius2,3,1)
		layout.addWidget(self.button_removebg,4,0)

		self.setLayout(layout)

		self.combo_method.addItems(['None','Uniform','Gaussian','Contrast','Median'])
		self.combo_method.setCurrentIndex(self.method)
		self.combo_method.currentIndexChanged.connect(self.update_method)

	def update_radius1(self):
		self.radius1 = float(self.le_radius1.text())

	def update_radius2(self):
		self.radius2 = float(self.le_radius2.text())

	def update_method(self,i):
		self.method = i

	def bg_filter(self,data):
		from scipy import ndimage as nd
		self.gui.set_status('Removing Background')

		filter_type = None
		if self.method == 1:
			filter_type = 'uniform'
		elif self.method == 2:
			filter_type = 'gaussian'
		elif self.method == 3:
			filter_type = 'minmax'
		elif self.method == 4:
			filter_type = 'median'

		wtime = 0
		wspace1 = float(self.le_radius1.text())
		wspace2 = float(self.le_radius2.text())

		if data.ndim == 2:
			shape1 = (wspace1,)*2
			shape2 = (wspace2,)*2
		else:
			shape1 = (wtime,wspace1,wspace1)
			shape2 = (wtime,wspace2,wspace2)

		f1 = None
		f2 = None
		if filter_type == 'uniform':
			f1 = nd.uniform_filter
			f2 = nd.uniform_filter
		elif filter_type == 'gaussian':
			f1 = nd.gaussian_filter
			f2 = nd.gaussian_filter
		elif filter_type == 'minmax':
			f1 = nd.maximum_filter
			f2 = nd.minimum_filter
		elif filter_type == 'median':
			f1 = nd.median_filter
			f2 = nd.median_filter

		if not f1 is None and not f2 is None:
			data = f1(data,shape1) - f2(data,shape2)

		return data

	def remove_bg(self):
		if self.gui.data.flag_movie:
			self.gui.data.movie = self.bg_filter(self.gui.data.movie.astype('float'))
			self.gui.docks['play'][1].update_frame()
			self.gui.docks['contrast'][1].update_image_contrast()
			self.method = 0
			self.combo_method.setCurrentIndex(self.method)
