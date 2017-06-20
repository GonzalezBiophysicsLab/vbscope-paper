from PyQt5.QtWidgets import QWidget, QSlider, QLabel, QGridLayout, QSizePolicy,QHBoxLayout
from PyQt5.QtCore import Qt

import numpy as np

class dock_contrast(QWidget):
	def __init__(self,parent):
		super(dock_contrast, self).__init__(parent)

		self.gui = parent

		### setup layout
		grid = QGridLayout()

		self.slider_ceiling = QSlider(Qt.Horizontal)
		self.slider_floor = QSlider(Qt.Horizontal)

		[ss.setMinimum(0.) for ss in [self.slider_ceiling,self.slider_floor]]
		[ss.setMaximum(10000.) for ss in [self.slider_ceiling,self.slider_floor]]
		self.slider_ceiling.setValue(10000.)
		self.slider_floor.setValue(0.)

		sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
		self.slider_ceiling.setSizePolicy(sizePolicy)
		self.slider_floor.setSizePolicy(sizePolicy)

		grid.addWidget(QLabel('Floor'),0,0)
		grid.addWidget(QLabel('Ceiling'),1,0)
		grid.addWidget(self.slider_floor,0,1)
		grid.addWidget(self.slider_ceiling,1,1)

		self.setLayout(grid)

		sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
		self.setSizePolicy(sizePolicy)

		### connect buttons
		self.slider_floor.valueChanged.connect(self.change_floor)
		self.slider_ceiling.valueChanged.connect(self.change_ceiling)
		self.slider_floor.sliderReleased.connect(self.update_image_contrast)
		self.slider_ceiling.sliderReleased.connect(self.update_image_contrast)

	def update_image_contrast(self):
		try:
			d = self.gui.data
			if d.flag_movie:
				xx = self.gui.prefs['contrast_scale']*((d.image_contrast / 100.) -.5)
				y = 1./(1.+np.exp(-xx))
				m = d.movie[d.current_frame]
				# m -= self.gui.docks['background'][1].calc_background(m)
				self.gui.plot.image.set_clim(np.percentile(m,y[0]*100.), np.percentile(m,y[1]*100.))

				# self.gui.plot.image.set_clim(np.percentile(d.movie[d.current_frame],y[0]*100.),
					# np.percentile(d.movie[d.current_frame],y[1]*100.))
				self.gui.plot.draw()
		except:
			pass

	def change_floor(self,v):
		vv = float(v)/100.
		self.gui.data.image_contrast[0] = vv
		if vv >= self.gui.data.image_contrast[1]:
			self.gui.data.image_contrast[0] = self.gui.data.image_contrast[1] - 1
			self.slider_floor.setValue(self.gui.data.image_contrast[0])
		self.update_image_contrast()

	def change_ceiling(self,v):
		vv = float(v)/100.
		self.gui.data.image_contrast[1] = vv
		if vv <= self.gui.data.image_contrast[0]:
			self.gui.data.image_contrast[1] = self.gui.data.image_contrast[0] + 1
			self.slider_ceiling.setValue(self.gui.data.image_contrast[1])
		self.update_image_contrast()
