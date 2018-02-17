from PyQt5.QtWidgets import QMainWindow,QWidget,QHBoxLayout,QTableWidget,QSizePolicy,QTableWidgetItem
from PyQt5.QtCore import Qt

import numpy as np
import multiprocessing as mp
import time
from matplotlib.pyplot import cm


default = {
	'computer_ncpu':mp.cpu_count(),

	'ui_bgcolor':'white',
	'ui_fontcolor':'grey',
	'ui_height':500,
	'ui_version':0.1,
	'ui_width':700
}


class preferences(QMainWindow):
	'''
	feed me a gui
	'''

	def __init__(self,gui):
		super(QMainWindow,self).__init__()

		self.gui = gui
		self.gui.prefs = default

		self.init_ui()
		self.init_table()

	def init_ui(self):

		self.viewer = QTableWidget()
		self.viewer.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)

		self.setCentralWidget(self.viewer)
		self.adjustSize()

	def edit(self,a):

		pref = self.viewer.verticalHeaderItem(a.row()).text()
		old = self.gui.prefs[pref]

		try:
			if type(old) is int:
				self.gui.prefs[pref] = int(a.text())
			elif type(old) is float:
				self.gui.prefs[pref] = float(a.text())
			elif type(old) is np.ndarray:
				self.gui.prefs[pref][a.column()] = float(a.text())
			elif type(old) is list:
				self.gui.prefs[pref][a.column()] = a.text()

			else:
				self.gui.prefs[pref] = a.text()

			## colormap safety
			if pref == 'plot_colormap':
				if not cm.__dict__.has_key(self.gui.prefs[pref]):
					self.gui.prefs[pref] = 'viridis'
				self.gui.plot.image.set_cmap(self.gui.prefs[pref])
				self.gui.plot.canvas.draw()

			self.gui.log('Pref: %s - %s > %s'%(pref,str(old),a.text()),True)

		except:
			pass

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
			self.gui.ui_update()
		except:
			pass

	def merge_dictionaries(self,newd,oldd):
		z = oldd.copy()
		z.update(newd)
		return z

	def combine_prefs(self,add_prefs_dictionary):
		self.gui.prefs = self.merge_dictionaries(add_prefs_dictionary,self.gui.prefs)
		self.update_table()

	def add_pref(self,key,value):
		try:
			self.gui.prefs[key] = value
			self.update_table
		except:
			pass
