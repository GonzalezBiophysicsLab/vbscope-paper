from PyQt5.QtWidgets import QMainWindow, QDockWidget, QAction, QMessageBox,QProgressDialog,QMessageBox,QShortcut
from PyQt5.QtCore import Qt, qInstallMessageHandler
from PyQt5.QtGui import QKeySequence

# def handler(msg_type, msg_log_context, msg_string):
	# pass
# qInstallMessageHandler(handler)

## Force Qt5 for matplotlib
try:
	import matplotlib
	matplotlib.use('Qt5Agg')
except:
	pass

from ui_log import logger
from ui_prefs import preferences

class gui(QMainWindow):
	'''
	UI Objects of Importance:
		* app
		* menubar
		* _log
		* prefs
		* docks

	Functions of Importance:
		* add_dock
		* load_movie
		* set_status
		* log
		* load - overload this
		* prefs.add_dictionary (for a dictionary)
		* prefs['new_pref_name'] = new_pref_value  (for one entry)

	Variables of Importance:
		* app_name
		* about_text
		* prefs
	'''
	## app - the main app
	## menubar - the menubar
	## log - logger
	## prefs - preferences
	## statusbar - status bar

	## add_dock - add a dock

	def __init__(self,app=None,main_widget=None):
		super(QMainWindow,self).__init__()
		self.app = app
		self.app_name = ""

		if not main_widget is None:
			self.setCentralWidget(main_widget)

		self.closeEvent = self.safe_close

		self.init_menus()
		self.init_docks()
		self.init_statusbar()
		self.init_shortcuts()

		self._log = logger()
		self.prefs = preferences(self)

		self.ui_update()
		self.show()

################################################################################

	def init_shortcuts(self):
		f12 = QShortcut(QKeySequence("F12"), self, self.full_screen)
		f1 = QShortcut(QKeySequence("F1"), self, self.open_main)
		for f in [f1,f12]:
			f.setContext(Qt.ApplicationShortcut)

	def init_statusbar(self):
		self.statusbar = self.statusBar()
		self.statusbar.showMessage('Initialized')

	def add_dock(self,name,title,widget,areas,loc):
		self.docks[name] = [QDockWidget(title, self), widget]
		if areas == 'lr':
			ar = Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
		elif areas == 'tb':
			ar = Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea
		self.docks[name][0].setAllowedAreas(ar)
		self.docks[name][0].setWidget(self.docks[name][1])
		if loc == 't':
			l = Qt.TopDockWidgetArea
		elif loc == 'b':
			l = Qt.BottomDockWidgetArea
		elif loc == 'l':
			l = Qt.LeftDockWidgetArea
		elif loc == 'r':
			l = Qt.RightDockWidgetArea
		self.addDockWidget(l, self.docks[name][0])

		try:
			self.prefs.add_dictionary(widget.default_prefs)

		except:
			pass

	def init_docks(self):
		# self.docks is a dictionary with [QDockWidget,Widget] for the docks
		self.docks = {}

	def init_menus(self):
		self.menubar = self.menuBar()
		self.menubar.setNativeMenuBar(False)

		### File
		self.menu_file = self.menubar.addMenu('File')

		file_load = QAction('Load', self, shortcut='Ctrl+O')
		file_load.triggered.connect(lambda e: self.load())

		file_log = QAction('Log', self,shortcut='F2')
		file_log.setShortcutContext(Qt.ApplicationShortcut)
		file_log.triggered.connect(self.open_log)

		file_prefs = QAction('Preferences', self,shortcut='F3')
		file_prefs.setShortcutContext(Qt.ApplicationShortcut)
		file_prefs.triggered.connect(self.open_preferences)

		self.about_text = ""
		file_about = QAction('About',self)
		file_about.triggered.connect(self.about)

		file_exit = QAction('Exit', self, shortcut='Ctrl+Q')
		# file_exit.triggered.connect(self.app.quit)
		file_exit.triggered.connect(self.close)

		for f in [file_load,file_log,file_prefs,file_about,file_exit]:
			self.menu_file.addAction(f)

################################################################################

	def load(self):
		'''
		Overload me
		'''
		pass

	def log(self,line,timestamp = False):
		self._log.log(line,timestamp)
		self.set_status(line)

	def about(self):
		QMessageBox.about(None,'About %s'%(self.app_name),self.about_text)

	def set_status(self,message=""):
		self.statusbar.showMessage(message)
		self.app.processEvents()

################################################################################

	def open_log(self):
		self._open_ui(self._log)

	def open_preferences(self):
		self._open_ui(self.prefs)

	def open_main(self):
		self._open_ui(self)

	def _open_ui(self,ui):
		try:
			if not ui.isVisible():
				ui.setVisible(True)
			ui.raise_()
		except:
			ui.show()
		ui.showNormal()
		ui.activateWindow()

	def ui_update(self):
		# self.resize(QDesktopWidget().availableGeometry(self).size() * 0.5)
		# self.menubar.setStyleSheet('background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 lightgray, stop:1 darkgray)')

		for s in [self,self._log,self.prefs]:
			s.setStyleSheet('''
			color:%s;
			background-color:%s;
			font-size: %spx;
		'''%(self.prefs['ui_fontcolor'],self.prefs['ui_bgcolor'],self.prefs['ui_fontsize']))

		self.resize(self.prefs['ui_width'],self.prefs['ui_height'])
		self.setWindowTitle(self.app_name)

	def quicksafe_load(self,fname):
		import numpy as np
		try:
			f = open(fname,'r')
			l = f.readline()
			f.close()
			if l.count(',') > 0:
				delim = ','
			else:
				delim = ' '

			f = open(fname,'r')
			d = []
			for line in f:
				d.append([float(n) for n in line.split(delim)])
			return np.array(d)
		except:
			return np.loadtxt(fname)

		# return np.loadtxt(fname,delimiter=delim)

################################################################################

	def full_screen(self):
		if self.isFullScreen():
			self.showNormal()
		else:
			self.showFullScreen()

	def unsafe_close(self,event):
		event.accept()

	def safe_close(self,event):
		# event.ignore()
		reply = QMessageBox.question(self,"Quit?","Are you sure you want to quit?",QMessageBox.Yes | QMessageBox.No)
		if reply == QMessageBox.Yes:
			# self.app.quit()
			self._log.close()
			self.prefs.close()
			event.accept()
		else:
			event.ignore()
