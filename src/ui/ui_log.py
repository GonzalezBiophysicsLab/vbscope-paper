from PyQt5.QtWidgets import QMainWindow, QWidget,QHBoxLayout,QVBoxLayout,QPushButton,QPlainTextEdit,QFileDialog

import time

class logger(QMainWindow):
	'''
	Use .log(string) to log a line
	Use .log(string,True) to timestamp it
	'''

	def __init__(self,gui=None):
		super(QMainWindow,self).__init__()

		self.init_ui()
		self.log('Initialized Log',True)

	def init_ui(self):
		self.textedit = QPlainTextEdit()
		self.textedit.setReadOnly(True)
		self.textedit.setStyleSheet('font:Courier')

		self.button_save = QPushButton("Save")
		self.button_save.clicked.connect(self.save)

		qw = QWidget()
		vb = QVBoxLayout()

		qwb = QWidget()
		hb = QHBoxLayout()
		hb.addWidget(self.button_save)
		hb.addStretch(1)
		qwb.setLayout(hb)

		vb.addWidget(self.textedit)
		vb.addWidget(qwb)
		qw.setLayout(vb)

		self.setCentralWidget(qw)
		self.resize(800,300)

	def log(self,line,timestamp = False):
		if timestamp:
			t = time.localtime()
			tstr = "%02d:%02d:%02d"%(t.tm_hour,t.tm_min,t.tm_sec)
			self.textedit.appendPlainText('(%s) %s'%(tstr,line))
		else:
			self.textedit.appendPlainText(str(line))

	def save(self):
		fname = QFileDialog.getSaveFileName(self, 'Export Log File', "_log.txt",'*.txt')
		if fname[0] != "":
			with open(fname[0],'w') as f:
				f.write(self.textedit.toPlainText())
				f.close()
