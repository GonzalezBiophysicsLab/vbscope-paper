from PyQt5.QtWidgets import QWidget, QSizePolicy, QTableWidget, QHBoxLayout, QTableWidgetItem
from PyQt5.QtCore import Qt

import numpy as np
import multiprocessing as mp

import matplotlib.pyplot as plt

last_update_date = '2017/11/14'

class dock_prefs(QWidget):
	def __init__(self,parent=None):
		super(dock_prefs, self).__init__()

		try:
			self.parent = parent
			self.gui = parent.gui
		except:
			self.gui = parent

		hbox = QHBoxLayout()
		self.viewer = QTableWidget()
		self.viewer.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
		# hbox.addStretch(1)
		hbox.addWidget(self.viewer)
		self.setLayout(hbox)
		self.adjustSize()

		# self.viewer.viewportEntered.connect(self.update_table)

		self.init_table()

	def edit(self,a):

		pref = self.viewer.verticalHeaderItem(a.row()).text()
		old = self.gui.prefs[pref]

		# try:
		if 1:
			if type(old) is int:
				# print 1
				self.gui.prefs[pref] = int(a.text())
			elif type(old) is float:
				# print 2
				self.gui.prefs[pref] = float(a.text())
				# print 3
			elif type(old) is np.ndarray:
				# print 5
				self.gui.prefs[pref][a.column()] = float(a.text())
			elif type(old) is list:
				self.gui.prefs[pref][a.column()] = a.text()
			else:
				# print 4
				self.gui.prefs[pref] = a.text()
		# except:
			# pass
		# print self.gui.prefs[pref]
		####
		if pref == 'color map':
			if not plt.cm.__dict__.has_key(self.gui.prefs[pref]):
				self.gui.prefs[pref] = 'viridis'
			self.gui.plot.image.set_cmap(self.gui.prefs[pref])
			self.gui.plot.canvas.draw()

		self.update_table()


	def init_table(self):
		self.viewer.setHorizontalHeaderLabels([ "Parameter", "Value"])
		self.viewer.itemChanged.connect(self.edit)
		self.update_table()
		self.viewer.show()


	def update_table(self):
		self.viewer.blockSignals(True)
		p = self.gui.prefs.items()
		p.sort()
		rows = len(p)

		columns = np.max([np.size(p[i][1]) for i in range(rows)])
		self.viewer.setRowCount(rows)
		self.viewer.setColumnCount(columns)

		labels = []
		for i in range(len(p)):
			labels.append(str(p[i][0]))
			s = np.size(p[i][1])

			if s == 1:
				v = [QTableWidgetItem(str(p[i][1]))]
			else:
				v = [QTableWidgetItem(str(p[i][1][j])) for j in range(s)]

			for j in range(len(v)):
				self.viewer.setItem(i,j,v[j])

		self.viewer.setVerticalHeaderLabels(labels)

		self.viewer.resizeColumnsToContents()
		self.viewer.resizeRowsToContents()
		self.viewer.blockSignals(False)

		try:
			self.parent.update()
		except:
			pass


default = {

'channel_wavelengths':np.array((570.,680.,488.,800.)),
'channel_colors':['green','red','blue','purple'],

'pixel_size':13300,
'magnification':60,
'binning':2,
'numerical_aperture':1.2,

'nsearch':3,
'nintegrate':7,
'clip border':5,

'color map':'Greys_r',#'viridis',

'tau':0.1,
'bleedthrough':np.array(((0.,0.05,0.,0.),(0.,0.,0.,0.),(0.,0.,0.,0.),(0.,0.,0.,0.))).flatten(),
'same_cutoff':1.,

'playback_fps':100,

'alignment_order':4,
'contrast_scale':20.,

'ml_psf_maxiters':1000,
'maxiterations':1000,
'threshold':1e-10,
'ncpu':mp.cpu_count(),
'nstates':4,

'downsample':1,
'snr_threshold':.5,
'pb_length':10,

'plotter_floor':0.2,
'plotter_nbins_contour':20,
'plotter_smoothx':0.5,
'plotter_smoothy':0.5,
'plotter_timeshift':0.0,
'plotter_floorcolor':'lightgoldenrodyellow',
'plotter_cmap':'rainbow',
'plotter_min_fret':-.5,
'plotter_max_fret':1.5,
'plotter_nbins_fret':41,
'plotter_min_time':0,
'plotter_max_time':100,
'plotter_nbins_time':100,
'plotter_syncpreframes':10,

'convert_flag':False,
'convert_c_lambda':[11.64,12.75],
'convert_em_gain':300,
'convert_offset':0,

'photobleaching_flag':True,
'synchronize_start_flag':False,

'hmm_nrestarts':4,
'hmm_sigmasmooth':False
}
