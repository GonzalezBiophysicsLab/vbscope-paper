from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection


from PyQt5.QtWidgets import QSizePolicy

import numpy as np

class plot_container():
	'''
	.f - figure
	.ax - axis
	.toolbar - MPL toolbar
	.image - pixel image
	.spots - line overlay w/ spot positions
	'''
	def __init__(self):

		# self.colorlist = ['lime','red','cyan','yellow','purple','k']
		self.colorlist_ordered = ['red','green','cyan','purple','k']
		self.colorlist = self.colorlist_ordered

		self.f,self.ax = plt.subplots(1)
		self.canvas = FigureCanvas(self.f)
		self.toolbar = NavigationToolbar(self.canvas,None)

		sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
		self.canvas.setSizePolicy(sizePolicy)

		self.image = self.ax.imshow(np.zeros((10,10),dtype='f'),extent=[0,10,0,10],cmap='viridis',interpolation='nearest',origin='lower')
		self.background = np.zeros_like(self.image.get_array(),dtype='f')

		self.spots = []

		self.ax.set_axis_off()
		self.ax.set_visible(False)

		self.f.subplots_adjust(left=.05,right=.95,top=.95,bottom=.05)
		self.canvas.draw()
		plt.close(self.f)

	def draw(self):
		self.ax.draw_artist(self.image)
		[self.ax.draw_artist(c) for c in self.ax.collections]
		self.canvas.update()
		self.canvas.flush_events()

	def clear_collections(self):
		for i in range(len(self.ax.collections)):
			self.ax.collections[0].remove()

	def scatter(self,x,y,radius=.66,color='red'):
		ps = [RegularPolygon([y[i]+.5,x[i]+.5],4,radius=radius) for i in range(len(x))]
		pc = PatchCollection(ps,alpha=.6,facecolor=color,edgecolor=color)
		self.ax.add_collection(pc)
		# self.canvas.draw()
