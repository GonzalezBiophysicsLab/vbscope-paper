from PyQt5.QtWidgets import QProgressDialog

class progressbar(QProgressDialog):
	'''
	prog = progressbar()
	prog.setRange(0,100)
	prog.show()
	self.gui.app.processEvents()

	import time
	for i in range(100):
		prog.setValue(i)
		self.app.processEvents()
		time.sleep(.5)
	'''

	def __init__(self):
		QProgressDialog.__init__(self)
		from PyQt5.Qt import QFont
		self.setFont(QFont('monospace'))

		self.setStyleSheet('''
			.QProgressDialog {
				border: 2px solid grey;
				border-radius: 5px;
			}
			.QProgressDialog::chunk {
				background-color: #05B8CC;
				width:20px;
			}
		''')
