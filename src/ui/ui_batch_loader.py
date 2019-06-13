from PyQt5.QtWidgets import QListWidget,QMainWindow,QWidget,QAbstractItemView,QFileDialog,QPushButton,QVBoxLayout,QHBoxLayout,QLabel

from PyQt5.QtCore import Qt

class deleting_list(QListWidget):
	def __init__(self):
		super(deleting_list,self).__init__()
		self.setSelectionMode(QAbstractItemView.ExtendedSelection)
		self.setDragEnabled(True)
		self.setDropIndicatorShown(True)
		self.setDragDropMode(QAbstractItemView.InternalMove)

	def keyPressEvent(self,event):
		if event.key() == Qt.Key_Backspace or event.key() == Qt.Key_Delete:
			sis = self.selectedItems()
			if len(sis) > 0:
				for si in sis:
					self.takeItem(self.row(si))
		QListWidget.keyPressEvent(self,event)
		event.accept()

	def load_files(self):
		fname = QFileDialog.getOpenFileNames(self,'Choose Files','./')
		if len(fname[0]) > 0:
			self.addItems(fname[0])


class gui_batch_loader(QMainWindow):
	def __init__(self,parent=None):
		super(QMainWindow,self).__init__(parent)
		self.ui = batch_loader(self)
		self.setCentralWidget(self.ui)
		self.show()

	def closeEvent(self,event):
		self.parent().activateWindow()
		self.parent().raise_()
		self.parent().setFocus()

class batch_loader(QWidget):
	def __init__(self,parent=None):
		super(QWidget,self).__init__(parent=parent)
		self.initialize()
		self.window = self.parent().parent()

	def initialize(self):
		tothbox = QHBoxLayout()

		widl1 = QWidget()
		vbox = QVBoxLayout()
		self.l1 = deleting_list()
		vbox.addWidget(QLabel('Trajectories'))
		vbox.addWidget(self.l1)
		hw = QWidget()
		hbox = QHBoxLayout()
		b1a = QPushButton('Add')
		b1c = QPushButton('Clear')
		hbox.addStretch()
		hbox.addWidget(b1a)
		hbox.addWidget(b1c)
		hw.setLayout(hbox)
		vbox.addWidget(hw)
		b1a.clicked.connect(self.l1.load_files)
		b1c.clicked.connect(self.l1.clear)
		widl1.setLayout(vbox)

		widl2 = QWidget()
		vbox = QVBoxLayout()
		self.l2 = deleting_list()
		vbox.addWidget(QLabel('Classes/Bleaching'))
		vbox.addWidget(self.l2)
		hw = QWidget()
		hbox = QHBoxLayout()
		b2a = QPushButton('Add')
		b2c = QPushButton('Clear')
		hbox.addStretch()
		hbox.addWidget(b2a)
		hbox.addWidget(b2c)
		hw.setLayout(hbox)
		vbox.addWidget(hw)
		b2a.clicked.connect(self.l2.load_files)
		b2c.clicked.connect(self.l2.clear)
		widl2.setLayout(vbox)

		vwid = QWidget()
		vboxsub = QVBoxLayout()
		ss = 'Bulk Import\nPick trajectories and matching class files\nfor bulk import.  Match corresponding files\nin the same row by dragging and dropping.\nRemove files with delete key or clear. If\nyou do not have class files, keep the class\npane empty. Click the load button to load.'
		bsub = QPushButton('Load')
		vboxsub.addWidget(QLabel(ss))
		vboxsub.addWidget(bsub)
		vwid.setLayout(vboxsub)
		bsub.clicked.connect(self.batch_load)

		tothbox.addWidget(widl1)
		tothbox.addWidget(widl2)
		tothbox.addWidget(vwid)

		self.setLayout(tothbox)

	def keyPressEvent(self,event):
		if event.key() == Qt.Key_Escape:
			self.window.open_preferences()
			return
		super(batch_loader,self).keyPressEvent(event)

	def get_lists(self):
		count1 = self.l1.count()
		count2 = self.l2.count()
		if count1 == 0:
			return None
		elif (count2 == 0) or (count1 == count2):
			ltraj = []
			lclass = []
			for i in range(count1):
				ltraj.append(self.l1.item(i).text())
				if count2 != 0:
					lclass.append(self.l2.item(i).text())
				else:
					lclass.append(None)
			return ltraj,lclass

	def batch_load(self):
		ls = self.get_lists()
		if not ls is None:
			ltraj,lclass = ls
			self.window.load_batch(ltraj,lclass)
