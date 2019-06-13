from PyQt5.QtWidgets import QWidget, QSizePolicy, QPushButton, QSpinBox, QLabel, QHBoxLayout, QFileDialog,QGridLayout
from PyQt5.QtCore import Qt, QTimer
from PyQt5.Qt import QFont

import numpy as np

from ..ui import progressbar

class dock_render(QWidget):
	def __init__(self,parent=None):
		super(dock_render, self).__init__(parent)

		self.gui = parent

		#### Render Movie
		box_render = QGridLayout()
		self.button_render = QPushButton('Render')
		self.spin_start = QSpinBox()
		self.spin_end = QSpinBox()

		sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
		for s in [self.spin_start,self.spin_end]:
			# s.setSizePolicy(sizePolicy)
			s.setMinimum(1)
			s.setMaximum(1)
			s.setValue(1)


		box_render.addWidget(QLabel("Start Frame"),0,0)
		box_render.addWidget(QLabel("End Frame"),0,1)
		box_render.addWidget(self.button_render,1,2)
		box_render.addWidget(self.spin_start,1,0)
		box_render.addWidget(self.spin_end,1,1)
		self.setLayout(box_render)

		self.button_render.clicked.connect(self.render)

		self.flag_rendering = False

	def cancel_render(self):
		self.flag_rendering = False

	def render(self):
		f0 = self.spin_start.value() - 1
		f1 = self.spin_end.value() - 1
		nf = f1 - f0
		if f1 < 1:
			self.gui.log('Render: Failed - end < start frame',True)
			return

		oname = QFileDialog.getSaveFileName(self, 'Render Movie', 'Movie.mp4','*.mp4')
		if oname[0] != "":
			self.gui.log('Render: Begin %d to %d; %d FPS'%(f0,f1,self.gui.prefs['render_fps']),True)
			prog = progressbar()
			prog.setRange(0,nf)
			prog.setWindowTitle('Rendering Frames')
			prog.setLabelText('Progress')
			self.flag_rendering = True
			prog.canceled.connect(self.cancel_render)
			prog.show()

			from matplotlib import animation
			ffmpegwriter = animation.writers[self.gui.prefs['render_renderer']]
			metadata = dict(title=self.gui.prefs['render_title'],artist=self.gui.prefs['render_artist'],comment=str(self.gui.data.filename))

			writer = ffmpegwriter(fps=self.gui.prefs['render_fps'],metadata=metadata,codec=self.gui.prefs['render_codec'])
			cf = self.gui.data.current_frame
			self.gui.data.current_frame = f0
			self.gui.docks['play'][1].update_frame()
			try:
				with writer.saving(self.gui.plot.f,oname[0],nf):
					for i in range(nf):
						if not self.flag_rendering:
							break
						self.gui.data.current_frame += 1
						self.gui.docks['play'][1].update_frame()
						writer.grab_frame()
						prog.setValue(i)
						self.gui.app.processEvents()
			except:
				writer = None
			self.gui.data.current_frame = cf
			self.gui.docks['play'][1].update_frame()
			self.gui.log('Render: Completed',True)
