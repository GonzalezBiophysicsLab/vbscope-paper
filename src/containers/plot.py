from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from matplotlib.widgets import  RectangleSelector


from PyQt5.QtWidgets import QSizePolicy,QVBoxLayout,QWidget

import numpy as np

class image_plot_container():
	'''
	.f - figure
	.ax - axis
	.toolbar - MPL toolbar
	.image - pixel image
	.spots - line overlay w/ spot positions
	'''
	def __init__(self):

		self.f,self.ax = plt.subplots(1)
		self.canvas = FigureCanvas(self.f)
		self.toolbar = NavigationToolbar(self.canvas,None)
		self.toolbar.actions()[0].triggered.connect(self.home_fxn)

		sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
		self.canvas.setSizePolicy(sizePolicy)

		self.image = self.ax.imshow(np.zeros((10,10),dtype='f'),extent=[0,10,0,10],cmap='Greys_r',interpolation='nearest',origin='lower')
		self.background = np.zeros_like(self.image.get_array(),dtype='f')

		self.spots = []

		self.ax.set_axis_off()
		self.ax.set_visible(False)

		self.setup_rectangle()

		self.f.subplots_adjust(left=.05,right=.95,top=.95,bottom=.05)
		self.canvas.draw()
		plt.close(self.f)

	def draw(self):
		self.ax.draw_artist(self.image)
		[self.ax.draw_artist(c) for c in self.ax.collections]
		if self.rectangle.visible:
			for artist in self.rectangle.artists:
				self.ax.draw_artist(artist)
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

	def setup_rectangle(self):
		def on_select(eclick, erelease):
			pass
		self.rectangle = None
		self.rectangle = RectangleSelector(self.ax, on_select, drawtype='box',useblit=False,button=[1,3],spancoords='pixels',interactive=True,rectprops = dict(facecolor='red', edgecolor = 'red', alpha=0.2, fill=True))

	def home_fxn(self):
		self.ax.autoscale()
		ny,nx = self.image.get_array().shape
		self.image.set_extent([0,nx,ny,0])
		# sizePolicy = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
		sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
		self.canvas.setSizePolicy(sizePolicy)
		self.draw()

	def remove_rectangle(self):
		for artist in self.rectangle.artists:
			artist.set_visible(False)
		self.rectangle.update()

class popup_plot_container(QWidget):
	def __init__(self,nplots=1):
		super(QWidget,self).__init__()

		self.nplots = nplots

		self.f,self.ax = plt.subplots(nplots,sharex=True)
		if not type(self.ax) is np.ndarray:
			self.ax = np.array([self.ax])
		self.canvas = FigureCanvas(self.f)
		self.toolbar = NavigationToolbar(self.canvas,None)

		# sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
		sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

		self.canvas.setSizePolicy(sizePolicy)
		self.f.set_dpi(self.f.get_dpi()/self.canvas.devicePixelRatio())
		self.fix_ax()

		self.canvas.draw()
		plt.close(self.f)

		layout = QVBoxLayout()
		layout.addWidget(self.canvas)
		layout.addWidget(self.toolbar)
		# layout.addStretch(1)
		self.setLayout(layout)
		self.f.tight_layout()

	def fix_ax(self):
		offset = .08
		offset2 = 0.14
		self.f.subplots_adjust(left=offset2,right=1.-offset,top=1.-offset,bottom=offset2)
		for aa in self.ax:
			aa.tick_params(labelsize=12./self.canvas.devicePixelRatio(),axis='both',direction='in',width=1.0/self.canvas.devicePixelRatio(),length=4./self.canvas.devicePixelRatio())

			aa.tick_params(axis='both', which='major', labelsize=12./self.canvas.devicePixelRatio())
			aa.format_coord = lambda x, y: ''


	def clf(self):
		self.f.clf()
		self.ax = np.array([self.f.add_subplot(self.nplots,1,i+1) for i in range(self.nplots)])
		self.fix_ax()

	def resizeEvent(self,event):
		self.fix_ax()
