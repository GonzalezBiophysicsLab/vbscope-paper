from PyQt5.QtWidgets import QApplication, QColumnView, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QMainWindow,QMessageBox,QSizePolicy
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QItemSelection

class dock_tagviewer(QWidget):
	def __init__(self,parent=None):
		super(dock_tagviewer, self).__init__(parent)

		self.gui = parent
		# self.gui.data.metadata
		self.initialize()

	def initialize(self):
		# self.setWindowTitle('HDF5 SMD Viewer')
		# self.setMinimumSize(800,0)

		self.viewer = QColumnView()#QTreeView()#QListView()
		self.viewer.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
		self.viewer.setResizeGripsVisible(True)

		vbox = QVBoxLayout()
		vbox.addWidget(self.viewer)
		vbox.addStretch(1)
		self.setLayout(vbox)

		self.init_model()

	def init_model(self):
		self.new_model(self.gui.data.metadata)
		if not self.model is None:
			self.viewer.setModel(self.model)
			self.show()
			self.adjustSize()
			self.parent().adjustSize()

	def new_model(self,filename):
		def add_group(l0,si):
			## Catch a weird thing
			if type(l0) is list:
				for i in range(len(l0)):
					si = add_group(l0[i],si)

			for i in range(len(list(l0.keys()))):
				key = list(l0.keys())[i]
				si_k = QStandardItem(key)
				si_k.setEditable(False)

				value = list(l0.values())[i]
				if type(value) is tuple or type(value) is list:
					value = str(value)

				if type(value) is dict:
					si_k = add_group(value,si_k)
				else:
					si_v = QStandardItem(value)
					si_v.setEditable(False)
					si_k.setChild(0,si_v)

				si.setChild(i,si_k)

			return si

		self.model = QStandardItemModel(self.viewer)
		# for i in range(len(self.gui.data.metadata)):
		if len(self.gui.data.metadata) > 1:
			for i in range(2):
				l = self.gui.data.metadata[i]
				dataset = QStandardItem(l[0])
				dataset.setEditable(False)
				add_group(l[1],dataset)
				dataset.sortChildren(0)
				self.model.appendRow(dataset)
