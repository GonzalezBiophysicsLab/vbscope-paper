import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget, QTextEdit, QMenuBar,  QAction, QMessageBox, QActionGroup
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class popplot_widget(QWidget):
	def __init__(self,nax1=1,nax2=1,toolbar=True,*args,**kwargs):
		super(QWidget,self).__init__(*args,**kwargs)

		## Create Figure
		self.fig, self.ax = plt.subplots(nax1, nax2, figsize = (4.0/QPixmap().devicePixelRatio(), 2.5/QPixmap().devicePixelRatio()))
		self.canvas = FigureCanvas(self.fig)
		self.fig.set_dpi(self.fig.get_dpi()/self.canvas.devicePixelRatio())
		self.canvas.draw()
		plt.close(self.fig)

		## Main Interface
		qvb = QVBoxLayout()
		qvb.addWidget(self.canvas)
		if toolbar:
			from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
			self.toolbar = NavigationToolbar(self.canvas,None)
			qvb.addWidget(self.toolbar)
		self.setLayout(qvb)

		## Setup Menu
		self.menubar = QMenuBar(self)
		self.menubar.setNativeMenuBar(False)
		self.layout().setMenuBar(self.menubar)
		self.add_menu_function('Close',self.close)

	def add_menu_function(self,title,function):
		action = QAction(title,self)
		action.triggered.connect(function)
		self.menubar.addAction(action)

	def add_menu_group(self,title,options_list,exclusive=False):
		'''
		self.group = self.add_menu_group('testing',['1','2','3','4'],False)
		for action in self.group.actions():
			print(action.text(),action.isChecked())
		'''
		group_menu = self.menubar.addMenu(title)
		group = QActionGroup(self)
		group.setExclusive(exclusive)
		for option in options_list:
			action = QAction(option,self)
			action.setCheckable(True)
			action.setChecked(True)
			group.addAction(action)
			group_menu.addAction(action)
		return group

if __name__ == '__main__':

	## View data
	import sys
	app = QApplication(sys.argv)
	view = popplot_widget()
	view.show()

	sys.exit(app.exec_())
