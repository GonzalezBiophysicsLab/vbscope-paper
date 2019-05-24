from PyQt5.QtWidgets import QListWidget,QMainWindow,QWidget,QAbstractItemView,QFileDialog,QPushButton,QVBoxLayout,QHBoxLayout,QLabel,QFormLayout,QLineEdit,QComboBox,QGridLayout
from PyQt5.QtCore import Qt
# from PyQt5.QtGui import QDoubleValidator, QIntValidator, QPixmap

import numpy as np

class gui_classifier(QMainWindow):
	def __init__(self,parent=None):
		super(QMainWindow,self).__init__(parent)
		self.ui = classifier(self)
		self.setCentralWidget(self.ui)
		self.show()

	def closeEvent(self,event):
		if not self.parent() is None:
			self.parent().raise_()
			self.parent().setFocus()

class classifier(QWidget):
	def __init__(self,parent=None):
		super(QWidget,self).__init__(parent=parent)
		self.gui= self.parent().parent()
		self.initialize()

	def initialize(self):

		self.selection_indices = None
		self.le_selection = QLineEdit('all')

		self.combo_action = QComboBox()
		self.label_info = QLabel()

		self.button_select = QPushButton('Select')
		self.button_process = QPushButton('Process')

		#####################################
		#####################################

		qgl = QGridLayout()
		qgl.addWidget(QLabel("Selection (pre,post,length,class,all,none,and,or)"),0,0)
		qgl.addWidget(self.le_selection,1,0)
		qgl.addWidget(self.button_select,1,1)
		qgl.addWidget(self.label_info,2,0)
		qgl.addWidget(self.combo_action,3,0)
		qgl.addWidget(self.button_process,3,1)
		self.setLayout(qgl)

		#####################################
		#####################################

		combo_options = ['Nothing','Remove Traces']
		[combo_options.append('Classify %d'%(i)) for i in range(10)]
		self.combo_action.addItems(combo_options)
		self.combo_action.setCurrentIndex(0)

		self.button_select.clicked.connect(self.make_selection)
		# self.combo_action.currentIndexChanged.connect(self.set_action)
		self.button_process.clicked.connect(self.process_selection)

	def keyPressEvent(self,event):
		if event.key() == Qt.Key_Escape:
			self.gui.open_preferences()
			return
		super(classifier,self).keyPressEvent(event)

	def make_selection(self):
		s = self.le_selection.text()
		success,inds = self.parse_selection(s)
		self.gui.log('Selection %s %s'%(success,s))
		if success:
			self.selection_indices = inds
			self.label_info.setText('Selected %d traces'%(inds.sum()))
		else:
			self.selection_indices = None
			self.label_info.setText('Selection failed')

	def process_selection(self):
		if self.selection_indices is None:
			return
		new_classes = np.copy(self.gui.data.class_list)
		keep = np.ones(new_classes.size,dtype='bool')
		ci = self.combo_action.currentIndex()
		inds = self.selection_indices
		if ci == 1: ## remove
			keep[self.selection_indices] = False
		elif ci > 1: ## class = ci-2
			new_classes[self.selection_indices] = ci-2

		old = self.gui.data.class_list.copy()
		self.gui.data.class_list = new_classes.copy()

		if not np.all(keep):
			if self.gui.data.remove_traces(keep):
				self.gui.plot.initialize_plots()
				self.gui.initialize_sliders()
				msg = "Filtered traces: kept %d out of %d = %f"%(keep.sum(),keep.size,keep.sum()/float(keep.size))
				self.gui.log(msg,True)
				self.gui.update_display_traces()
				self.selection_indices = None
			else:
				self.gui.data.class_list = old.copy()
				self.gui.update_display_traces()

	def parse_selection(self,s):
		ss = s.lower()
		ss = ss.split(' ')

		## Split up the phrases
		combines = {'and':np.bitwise_and, 'or':np.bitwise_or}
		for i in range(len(ss)):
			for c in combines.keys():
				if ss[i] == c:
					try:
						left = ' '.join(ss[:i])
						right = ' '.join(ss[i+1:])
					except:
						return False,None
					l = self.parse_selection(left)
					r = self.parse_selection(right)
					if l[0] and r[0]:
						try:
							return True,combines[c](l[1],r[1])
						except:
							self.gui.log('Failed %s %s %s'%(str(combines[c]),l[1],r[1]))
					return False,None

		## Process the phrases
		params = {
			'pre':'self.gui.data.pre_list',
			'post':'self.gui.data.pb_list',
			'length':'self.gui.data.pb_list - self.gui.data.pre_list',
			'class':'self.gui.data.class_list',
			'all':'np.ones(self.gui.data.pb_list.size,dtype=\'bool\')',
			'none':'np.zeros(self.gui.data.pb_list.size,dtype=\'bool\')',
		}

		for i in range(len(ss)):
			for p in params.keys():
				if ss[i] == p:
					ss[i] = params[p]
		ss = ' '.join(ss)

		try:
			return True,eval(ss)
		except:
			self.gui.log('Failed %s'%(ss))
			return False,None

if __name__ == '__main__':
	import sys
	from PyQt5.QtWidgets import QApplication
	app = QApplication([])
	app.setStyle('fusion')
	g = gui_classifier(parent=None)
	sys.exit(app.exec_())
