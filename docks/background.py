from PyQt5.QtWidgets import QWidget, QSizePolicy, QPushButton,QGridLayout,QComboBox,QLineEdit,QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator

import numpy as np

from scipy.ndimage import median_filter,gaussian_filter,minimum_filter

class dock_background(QWidget):
	def __init__(self,parent=None):
		super(dock_background, self).__init__(parent)
		self.gui = parent

		self.flag_showing = False

		self.radius1 = 5.
		self.radius2 = 0.5

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
		self.setLayout(layout)

		self.button_preview.clicked.connect(self.preview)

		self.method = 0
		self.combo_method.setCurrentIndex(self.method)
		self.combo_method.currentIndexChanged.connect(self.update_method)

		self.combo_method.addItems(['None','Minimum','Median'])
		self.combo_method.setCurrentIndex(0)

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
			return gaussian_filter(minimum_filter(image,self.radius1),self.radius2)
		elif self.method == 2:#'Median':
			return gaussian_filter(median_filter(image,int(self.radius1)),self.radius2)
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