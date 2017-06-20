from PyQt5.QtWidgets import QWidget, QSizePolicy, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt

import numpy as np

class dock_rotate(QWidget):
	def __init__(self,parent=None):
		super(dock_rotate, self).__init__(parent)

		self.gui = parent

		self.ccw = QPushButton("90 CCW")
		self.cw = QPushButton("90 CW")
		self.fx = QPushButton("Flip X")
		self.fy = QPushButton("Flip Y")

		layout = QHBoxLayout()
		layout.addWidget(self.ccw)
		layout.addWidget(self.cw)
		layout.addWidget(self.fx)
		layout.addWidget(self.fy)
		self.setLayout(layout)

		self.ccw.clicked.connect(self.rotate_ccw)
		self.cw.clicked.connect(self.rotate_cw)
		self.fx.clicked.connect(self.flip_x)
		self.fy.clicked.connect(self.flip_y)

	def rotate_ccw(self):
		if self.gui.data.flag_movie:
			self.gui.data.movie = np.rot90(self.gui.data.movie,k=3,axes=np.array((1,2)))
			self.gui.plot.image.set_array(np.rot90(self.gui.plot.image.get_array(),k=3))
			self.gui.plot.draw()
	def rotate_cw(self):
		if self.gui.data.flag_movie:
			self.gui.data.movie = np.rot90(self.gui.data.movie,k=1,axes=np.array((1,2)))
			self.gui.plot.image.set_array(np.rot90(self.gui.plot.image.get_array(),k=1))
			self.gui.plot.draw()
	def flip_x(self):
		if self.gui.data.flag_movie:
			self.gui.data.movie = self.gui.data.movie[:,:,::-1]
			self.gui.plot.image.set_array(self.gui.plot.image.get_array()[:,::-1])
			self.gui.plot.draw()
	def flip_y(self):
		if self.gui.data.flag_movie:
			self.gui.data.movie = self.gui.data.movie[:,::-1,:]
			self.gui.plot.image.set_array(self.gui.plot.image.get_array()[::-1])
			self.gui.plot.draw()
