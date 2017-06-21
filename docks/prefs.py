from PyQt5.QtWidgets import QWidget, QSizePolicy, QTableWidget, QHBoxLayout, QTableWidgetItem
from PyQt5.QtCore import Qt

import numpy as np
import matplotlib.pyplot as plt

class dock_prefs(QWidget):
	def __init__(self,parent=None):
		super(dock_prefs, self).__init__(parent)

		self.gui = parent

		hbox = QHBoxLayout()
		self.viewer = QTableWidget()
		self.viewer.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
		hbox.addStretch(1)
		hbox.addWidget(self.viewer)
		self.setLayout(hbox)
		self.adjustSize()

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
