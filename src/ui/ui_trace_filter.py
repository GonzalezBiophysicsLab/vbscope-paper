from PyQt5.QtWidgets import QListWidget,QMainWindow,QWidget,QAbstractItemView,QFileDialog,QPushButton,QVBoxLayout,QHBoxLayout,QLabel,QFormLayout,QLineEdit,QComboBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QPixmap

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import matplotlib.pyplot as plt
import numpy as np

class gui_trace_filter(QMainWindow):
	def __init__(self,parent=None):
		super(QMainWindow,self).__init__(parent)
		self.ui = trace_filter(self)
		self.setCentralWidget(self.ui)
		self.show()

	def closeEvent(self,event):
		if not self.parent() is None:
			self.parent().activateWindow()
			self.parent().raise_()
			self.parent().setFocus()

class trace_filter(QWidget):
	def __init__(self,parent=None):
		super(QWidget,self).__init__(parent=parent)
		self.initialize()
		self.window = self.parent().parent()

	def initialize(self):
		self.models_lists = [None]*5

		self.le_low_sbr = QLineEdit()
		self.le_high_sbr = QLineEdit()
		self.le_min_frames = QLineEdit()
		self.le_skip_frames = QLineEdit()
		self.combo_data = QComboBox()
		self.combo_data.addItems(['Donor','Acceptor','Donor+Acceptor'])

		dv = QDoubleValidator(0.1, 1000, 2)
		dv.setNotation(QDoubleValidator.StandardNotation)
		iv = QIntValidator(0, 10000)
		[le.setValidator(dv) for le in [self.le_low_sbr,self.le_high_sbr]]
		[le.setValidator(iv) for le in [self.le_min_frames,self.le_skip_frames]]
		self.reset_defaults()

		self.label_proportions = QLabel('[0, 0, 0, 0, 0]')

		self.fig, self.ax = plt.subplots(1, figsize = (4.0/QPixmap().devicePixelRatio(), 2.5/QPixmap().devicePixelRatio()),sharex=True)
		self.canvas = FigureCanvas(self.fig)
		self.toolbar = NavigationToolbar(self.canvas,None)
		self.fig.set_dpi(self.fig.get_dpi()/self.canvas.devicePixelRatio())
		self.canvas.draw()
		plt.close(self.fig)

		combo_options = ['Nothing','Remove Traces']
		[combo_options.append('Classify %d'%(i)) for i in range(10)]
		self.class_actions = []
		for i in range(5):
			self.class_actions.append(QComboBox())
			self.class_actions[i].addItems(combo_options)
			self.class_actions[i].setCurrentIndex(0)
		self.model_labels = ['Dead','Low SBR, Bleach', 'High SBR, Bleach','Low SBR, No bleach', 'High SBR, No bleach']

		self.button_defaults = QPushButton('Defaults')
		self.button_calculate = QPushButton('Calculate')
		self.button_process = QPushButton('Process')

		#####################################
		#####################################

		wid1 = QWidget()
		vbox = QVBoxLayout()
		vbox.addWidget(QLabel('Parameters'))
		qw = QWidget()
		fbox = QFormLayout()
		fbox.addRow("Low SBR",self.le_low_sbr)
		fbox.addRow("High SBR",self.le_high_sbr)
		fbox.addRow("Min. frames (dead)",self.le_min_frames)
		fbox.addRow("Skip frames (start)",self.le_skip_frames)
		fbox.addRow("Source Data",self.combo_data)
		qw.setLayout(fbox)
		vbox.addWidget(qw)

		qw = QWidget()
		hbox = QHBoxLayout()
		hbox.addWidget(self.button_defaults)
		hbox.addWidget(self.button_calculate)
		qw.setLayout(hbox)
		vbox.addWidget(qw)
		vbox.addStretch(1)
		wid1.setLayout(vbox)

		#####################################
		#####################################

		wid2 = QWidget()
		vbox = QVBoxLayout()
		vbox.addWidget(QLabel('Classifications'))
		vbox.addWidget(self.canvas)
		vbox.addWidget(self.toolbar)
		vbox.addWidget(self.label_proportions)
		vbox.addStretch(1)
		wid2.setLayout(vbox)

		#####################################
		#####################################

		wid3 = QWidget()
		vbox = QVBoxLayout()
		qw = QWidget()
		fbox = QFormLayout()
		for i in range(len(self.model_labels)):
			fbox.addRow(self.model_labels[i],self.class_actions[i])
		qw.setLayout(fbox)
		vbox.addWidget(qw)
		vbox.addWidget(self.button_process)
		wid3.setLayout(vbox)

		#####################################
		#####################################

		tothbox = QHBoxLayout()
		tothbox.addWidget(wid1)
		tothbox.addWidget(wid2)
		tothbox.addWidget(wid3)
		self.setLayout(tothbox)

		#####################################
		#####################################

		self.button_defaults.clicked.connect(self.reset_defaults)
		self.button_calculate.clicked.connect(self.calculate_model)
		self.button_process.clicked.connect(self.process_filters)

	def keyPressEvent(self,event):
		if event.key() == Qt.Key_Escape:
			self.window.open_preferences()
			return
		super(trace_filter,self).keyPressEvent(event)

	def reset_defaults(self):
		self.le_low_sbr.setText('2.0')
		self.le_high_sbr.setText('5.0')
		self.le_min_frames.setText('10')
		self.le_skip_frames.setText('0')
		self.combo_data.setCurrentIndex(2)

	def calculate_model(self):
		self.window.set_status('Compiling...')
		from ..supporting.trace_model_selection import model_select_many
		self.window.set_status('')
		if self.window.data.d is None:
			self.window.log('Load data first')
			return
		self.window.set_status('Running')
		low_sbr = float(self.le_low_sbr.text())
		high_sbr = float(self.le_high_sbr.text())
		min_frames = int(self.le_min_frames.text())
		skip_frames = int(self.le_skip_frames.text())
		mode = self.combo_data.currentIndex()
		mode_text = self.combo_data.currentText()

		if mode == 2:
			self.d = self.window.data.d[:,0] + self.window.data.d[:,1]
		else:
			self.d = self.window.data.d[:,mode]

		if skip_frames >= self.d.shape[0]:
			self.window.log('Too many skipped frames')
			return

		self.window.log('Running Filter Calculation: %d, %d, %i, %i, %s'%(low_sbr,high_sbr,min_frames,skip_frames,mode_text))

		self.probs = model_select_many(self.d[skip_frames:],low_sbr,high_sbr,min_frames)
		self.window.set_status('Done')
		nums = [int(np.sum(self.probs.argmax(1)==i)) for i in range(self.probs.shape[1])]
		self.label_proportions.setText(str(nums))

		pmax = self.probs.argmax(1)
		self.model_lists = [None]*5
		self.model_lists[0] = np.nonzero(np.bitwise_or(pmax == 0, pmax==3))[0] ## dead, any SBR
		self.model_lists[1] = np.nonzero(pmax == 1)[0] ## Bleaching, Low SBR
		self.model_lists[2] = np.nonzero(pmax == 4)[0] ## Bleaching, High SBR
		self.model_lists[3] = np.nonzero(pmax == 2)[0] ## No bleaching, Low SBR
		self.model_lists[4] = np.nonzero(pmax == 5)[0] ## No bleaching, High SBR

		## Plots
		self.window.set_status('Plotting')
		self.ax.cla()
		xmin = self.d.min()
		xmax = self.d.max()
		delta = xmax-xmin
		xmin -= delta*.05
		xmax += delta*.05

		colors = ['gray','blue','red','green','orange']
		for i in range(len(colors)):
			ml = self.model_lists[i]
			label = self.model_labels[i]
			self.ax.hist(self.d[ml].flatten(), range=(xmin,xmax), bins=300, histtype='step', lw=1.2, log=True, alpha=.8, label=label, color=colors[i])

		self.ax.hist(self.d.flatten(), range=(xmin,xmax), bins=300, histtype='step', lw=1.2, log=True, alpha=.8, label='All',color='k',ls='--')
		self.ax.legend(loc=1,fontsize='xx-small')
		self.ax.set_ylabel('Number of Datapoints')
		self.ax.set_xlabel('Intensity')
		self.fig.tight_layout()
		self.canvas.draw()


	def process_filters(self):
		if np.any([ml is None for ml in self.model_lists]):
			self.window.set_status('Calculate first')
			return

		wipe_flag = False
		new_classes = np.copy(self.window.data.class_list)
		keep = np.ones(new_classes.size,dtype='bool')
		for i in range(len(self.model_lists)):
			ml = self.model_lists[i]
			ci = self.class_actions[i].currentIndex()

			if ci == 1: ## remove
				keep[ml] = False
				wipe_flag = True
			if ci > 1: ## class = ci-2
				new_classes[ml] = ci-2

		old = self.window.data.class_list.copy()
		self.window.data.class_list = new_classes.copy()
		if self.window.data.remove_traces(keep):
			self.window.plot.initialize_plots()
			self.window.initialize_sliders()
			msg = "Filtered traces: kept %d out of %d = %f"%(keep.sum(),keep.size,keep.sum()/float(keep.size))
			self.window.log(msg,True)
			self.window.update_display_traces()
		else:
			self.window.data.class_list = old.copy()

		if wipe_flag:
			self.model_lists = [None]*5
			self.ax.cla()
			self.label_proportions.setText("")


	# def get_lists(self):
	# 	return None
	# 	# count1 = self.l1.count()
	# 	# count2 = self.l2.count()
	# 	# if count1 == 0:
	# 	# 	return None
	# 	# elif (count2 == 0) or (count1 == count2):
	# 	# 	ltraj = []
	# 	# 	lclass = []
	# 	# 	for i in range(count1):
	# 	# 		ltraj.append(self.l1.item(i).text())
	# 	# 		if count2 != 0:
	# 	# 			lclass.append(self.l2.item(i).text())
	# 	# 		else:
	# 	# 			lclass.append(None)
	# 	# 	return ltraj,lclass

	# def batch_load(self):
	# 	ls = self.get_lists()
	# 	if not ls is None:
	# 		ltraj,lclass = ls
	# 		self.window.load_batch(ltraj,lclass)

# if __name__=="__main__":
# 	from PyQt5.QtWidgets import QApplication
# 	import sys
# 	app = QApplication([])
# 	gui = gui_trace_filter()
# 	sys.exit(app.exec_())
