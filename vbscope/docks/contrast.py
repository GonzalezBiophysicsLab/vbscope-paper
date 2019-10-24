from PyQt5.QtWidgets import QWidget, QSlider, QLabel, QGridLayout, QSizePolicy,QHBoxLayout,QLineEdit,QPushButton
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator, QIntValidator

from matplotlib.colors import PowerNorm
import numpy as np

class dock_contrast(QWidget):
	def __init__(self,parent):
		super(dock_contrast, self).__init__(parent)

		self.flag_first_time = True
		self.gui = parent

		### setup layout
		grid = QGridLayout()

		# self.slider_ceiling = QSlider(Qt.Horizontal)
		# self.slider_floor = QSlider(Qt.Horizontal)
		#
		# [ss.setMinimum(0.) for ss in [self.slider_ceiling,self.slider_floor]]
		# [ss.setMaximum(10000.) for ss in [self.slider_ceiling,self.slider_floor]]
		# self.slider_ceiling.setValue(10000.)
		# self.slider_floor.setValue(0.)
		#
		# sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
		# self.slider_ceiling.setSizePolicy(sizePolicy)
		# self.slider_floor.setSizePolicy(sizePolicy)

		self.le_floor = QLineEdit()
		self.le_ceiling = QLineEdit()
		self.le_log10gamma = QLineEdit()
		self.button_guess = QPushButton('Guess')

		dv = QDoubleValidator(-1e300, 1e300, 2)
		dv.setNotation(QDoubleValidator.StandardNotation)
		[le.setValidator(dv) for le in [self.le_floor,self.le_ceiling,self.le_log10gamma]]
		self.reset_defaults()
		self.connect_things()

		grid.addWidget(QLabel('Floor'),0,0)
		grid.addWidget(QLabel('Ceiling'),0,2)
		grid.addWidget(QLabel('log10(gamma)'),0,4)
		grid.addWidget(self.le_floor,0,1)
		grid.addWidget(self.le_ceiling,0,3)
		grid.addWidget(self.le_log10gamma,0,5)
		grid.addWidget(self.button_guess,0,6)

		self.setLayout(grid)

		sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
		self.setSizePolicy(sizePolicy)

	def connect_things(self):
		self.le_floor.editingFinished.connect(self.change_floor)
		self.le_ceiling.editingFinished.connect(self.change_ceiling)
		self.le_log10gamma.editingFinished.connect(self.change_gamma)
		self.button_guess.clicked.connect(self.guess_contrast)

	def reset_defaults(self):
		self.gui.data.image_contrast[0] = 0.
		self.gui.data.image_contrast[1] = 0.
		self.gamma = 1.
		self.update_les()

	def update_les(self):
		self.le_floor.setText(str(self.gui.data.image_contrast[0]))
		self.le_ceiling.setText(str(self.gui.data.image_contrast[1]))
		self.le_log10gamma.setText(str(np.round(np.log10(self.gamma),3)))
		self.clear_le_focus()

	def clear_le_focus(self):
		[le.clearFocus() for le in [self.le_floor,self.le_ceiling,self.le_log10gamma]]

	def update_image_contrast(self):
		norm = PowerNorm(self.gamma,*self.gui.data.image_contrast)
		self.gui.plot.image.set_norm(norm)
		self.gui.plot.canvas.draw()
		self.clear_le_focus()

	def guess_contrast(self):
		try:
			im = self.gui.plot.image.get_array()
		except:
			im = self.gui.data.movie[self.gui.data.current_frame]
		self.gui.data.image_contrast[0] = im.min()
		self.gui.data.image_contrast[1] = im.max()
		self.gamma = 10.**-.1
		self.update_les()
		self.update_image_contrast()

	def change_floor(self):
		floor = float(self.le_floor.text())
		if floor > self.gui.data.image_contrast[1]:
			floor = self.gui.data.image_contrast[0]
		self.gui.data.image_contrast[0] = floor
		self.update_les()
		self.update_image_contrast()

	def change_ceiling(self):
		ceiling = float(self.le_ceiling.text())
		if ceiling < self.gui.data.image_contrast[0]:
			ceiling = self.gui.data.image_contrast[1]
		self.gui.data.image_contrast[1] = ceiling
		self.update_les()
		self.update_image_contrast()

	def change_gamma(self):
		lg = float(self.le_log10gamma.text())

		if lg > 1.:
			lg = 1.
		elif lg < -2:
			lg = -2.

		self.gamma = 10.**lg
		self.update_les()
		self.update_image_contrast()

	def change_contrast(self,l,h,g):
		self.gui.data.image_contrast[0] = l
		self.gui.data.image_contrast[1] = h
		self.gui.data.gamma = g
		self.update_les()
		self.update_image_contrast()
