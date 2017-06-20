from PyQt5.QtWidgets import QWidget, QSizePolicy, QPushButton, QSlider, QLabel, QHBoxLayout, QShortcut
from PyQt5.QtCore import Qt, QTimer
from PyQt5.Qt import QFont

import numpy as np

class dock_play(QWidget):
	def __init__(self,parent=None):
		super(dock_play, self).__init__(parent)

		self.gui = parent
		self.flag_playing = False

		#### Playbar
		hbox_play = QHBoxLayout()
		self.button_bck = QPushButton(u'\u2190')
		self.button_play = QPushButton(u"\u25B6")
		self.button_fwd = QPushButton(u'\u2192')
		self.slider_frame = QSlider(Qt.Horizontal)

		self.label_framenumber = QLabel('0 / 0')
		self.label_framenumber.setFont(QFont('monospace'))
		self.label_framenumber.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

		sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
		self.slider_frame.setSizePolicy(sizePolicy)

		self.slider_frame.setMinimum(1)
		self.slider_frame.setMaximum(1)
		self.slider_frame.setValue(self.gui.data.current_frame+1)

		hbox_play.addWidget(self.button_bck)
		hbox_play.addWidget(self.button_play)
		hbox_play.addWidget(self.button_fwd)
		hbox_play.addWidget(self.slider_frame)
		hbox_play.addWidget(self.label_framenumber)
		self.setLayout(hbox_play)

		self.button_fwd.clicked.connect(self.skip_p1)
		self.button_bck.clicked.connect(self.skip_m1)
		self.button_play.clicked.connect(self.play_movie)
		self.timer_playing = QTimer()

		self.slider_frame.valueChanged.connect(self.update_frame_slider)

		self.init_shortcuts()

	def make_shortcut(self,key,fxn):
		qs = QShortcut(self)
		qs.setKey(key)
		qs.activated.connect(fxn)

	def init_shortcuts(self):
		self.make_shortcut(Qt.Key_Left,self.skip_m1)
		self.make_shortcut(Qt.Key_Right,self.skip_p1)
		self.make_shortcut(" ",self.toggle_play)

	def toggle_play(self):
		if self.flag_playing:
			self.stop_playing()
		else:
			self.play_movie()


	def update_frame_slider(self,v=None):
		if not v is None:
			self.jump(v - 1)
			self.update_label()
			self.update_frame()


	def jump(self,v):
		if self.gui.data.flag_movie:
			self.gui.data.current_frame = v
			if self.gui.data.current_frame < 0:
				self.gui.data.current_frame = 0
			elif self.gui.data.current_frame > self.gui.data.total_frames-1:
				self.gui.data.current_frame = self.gui.data.total_frames-1

	def skip_p1(self):
		if self.gui.data.flag_movie:
			self.jump(self.gui.data.current_frame + 1)
			self.update_label()
			self.update_frame()

	def skip_m1(self):
		if self.gui.data.flag_movie:
			self.jump(self.gui.data.current_frame - 1)
			self.update_label()
			self.update_frame()

	def update_label(self):
		self.label_framenumber.setText('%0*d / %d'%(len(str(self.gui.data.total_frames)),self.gui.data.current_frame+1,self.gui.data.total_frames))

	def update_frame(self):
		self.gui.docks['background'][1].update_background()
		self.gui.plot.image.set_data(self.gui.data.movie[self.gui.data.current_frame] - self.gui.data.background)
		self.gui.plot.draw()

	def stop_playing(self):
		self.timer_playing.stop()
		self.flag_playing = False
		self.button_play.setText(u'\u25B6')
		self.slider_frame.setValue(self.gui.data.current_frame +1)
		self.timer_playing = QTimer()

	def advance_frame(self):

		if self.gui.data.current_frame < self.gui.data.total_frames-1:
			self.gui.data.current_frame += 1
			self.update_frame()
			self.update_label()
			self.slider_frame.blockSignals(True)
			self.slider_frame.setValue(self.gui.data.current_frame+1)
			self.slider_frame.blockSignals(False)
		else:
			self.stop_playing()


	def play_movie(self):
		if not self.flag_playing and self.gui.data.flag_movie:
			self.timer_playing.timeout.connect(self.advance_frame)
			self.flag_playing = True
			self.button_play.setText('||')
			self.timer_playing.start(1./self.gui.prefs['playback_fps']*1000)
		else:
			self.stop_playing()
