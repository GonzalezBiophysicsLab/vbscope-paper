
from PyQt5.QtWidgets import QMainWindow,QWidget,QHBoxLayout,QSizePolicy,QLineEdit, QTableView, QVBoxLayout, QWidget, QApplication, QStyledItemDelegate, QDoubleSpinBox, QShortcut, QFileDialog, QHeaderView,QPushButton, QStyle, QStyleOptionButton
from PyQt5.QtCore import  QRegExp, QSortFilterProxyModel, Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QKeySequence, QColor, QPalette

import numpy as np
import multiprocessing as mp
import time
from matplotlib.pyplot import cm

default_prefs = {
	'computer_ncpu':mp.cpu_count(),
	'ui_bgcolor':'white',
	'ui_fontcolor':'grey',
	'ui_fontsize':12.0,
	'ui_height':500,
	'ui_width':700,
	'pref_precision':6
}

class QAlmostStandardItemModel(QStandardItemModel):
	def __init__(self,*args):
		super(QAlmostStandardItemModel,self).__init__(*args)
		self.old_value = None
		self.old_index = None

	def setData(self,*args):
		index,value = args[:2]
		self.old_value = self.data(index)
		self.old_index = index
		value = self.enforce_type(self.old_value,value)
		return super(QAlmostStandardItemModel,self).setData(*(index,value)+args[2:])

	def enforce_type(self,old,new):
		if not old is None:
			if type(old) != type(new):
				if not type(old) in [type('a'),str] and not type(new) in [type('a'),str]:
					print('ignoring',old,new)
					return old
		return new

	def ddata(self,*args):
		d = super(QAlmostStandardItemModel,self).data(*args)
		if type(d) is type('d'):
			if d.count(',') > 0 and (d.count('[') > 0 or d.count('(') > 0):
				return np.array(eval(d))
		return d

class QAlmostLineEdit(QLineEdit):
	def __init__(self,*args):
		super(QAlmostLineEdit,self).__init__(*args)
	def keyPressEvent(self,event):
		super(QAlmostLineEdit,self).keyPressEvent(event)

class precision_delegate(QStyledItemDelegate):
	def __init__(self,*args):
		super(precision_delegate,self).__init__(*args)
		self.precision = default_prefs['pref_precision']
		self.refocus = lambda : None

	def createEditor(self, parent, option, index):
		value = index.data()
		editor = super(precision_delegate,self).createEditor(parent,option,index)
		if isinstance(value, float):
			editor.setDecimals(self.precision)
		if isinstance(value,command_obj):
			value.fxn()
			self.refocus()
			return None
		return editor

	def paint(self,painter,option,index):
		if (index.column() == 1) and type(index.data()) is command_obj:
			button = QStyleOptionButton()
			button.text = "Execute"
			button.rect = option.rect
			button.features = QStyleOptionButton.Flat
			QApplication.style().drawControl(QStyle.CE_PushButton,button,painter)
		else:
			super(precision_delegate,self).paint(painter,option,index)

class command_obj(object): ## generic class to take anything you throw at it...
	def __init__(self,name,fxn,*args):
		self.name = name
		self.fxn = fxn
		self.args = args


class preferences(QWidget):
	def __init__(self,parent=None):
		super(preferences, self).__init__()
		self.gui = parent

		self.model = QAlmostStandardItemModel(0, 2, self)

		self.model.setHeaderData(0, Qt.Horizontal, "Name")
		self.model.setHeaderData(1, Qt.Horizontal, "Value")

		self.proxy_model = QSortFilterProxyModel()
		self.proxy_model.setDynamicSortFilter(True)

		self.proxy_view = QTableView()
		self.delegate = precision_delegate()
		self.delegate.refocus = self.refocus_le
		self.proxy_view.setItemDelegate(self.delegate)

		self.proxy_view.setModel(self.proxy_model)
		self.proxy_view.setSortingEnabled(True)
		self.proxy_view.horizontalHeader().setStretchLastSection(True)
		# self.proxy_view.horizontalHeader().setVisible(False)
		self.proxy_view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
		self.proxy_view.verticalHeader().setVisible(False)

		self.le_filter = QAlmostLineEdit()
		self.le_filter.textChanged.connect(self.filter_regex_changed)
		self.le_filter.setPlaceholderText("Filter / command")

		proxyLayout = QVBoxLayout()
		proxyLayout.addWidget(self.le_filter)
		proxyLayout.addWidget(self.proxy_view)
		self.setLayout(proxyLayout)

		self.setWindowTitle("Preferences")

		self.model.itemChanged.connect(self.edit)
		self.edit_callback = lambda: None

		self.add_dictionary(default_prefs)
		self.resize_columns()
		self.le_filter.setFocus()

		undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"),self)
		undo_shortcut.activated.connect(self.undo)

		self.commands = {'load pref':self.load_preferences,'save pref':self.save_preferences,'hello world':(lambda : print('hello world'))}
		self.update_commands()
	# 	# self.proxy_view.doubleClicked.connect(self.clicked)
	# 	# self.proxy_view.activated.connect(self.clicked)
	#
	# def clicked(self,index):
	# 	clicksign = 'Execute'
	# 	print('asdf')
	# 	if index.column() == 1:
	# 		if self.proxy_model.itemData(self.proxy_model.index(index.row(),1))[0] == clicksign:
	# 			co = self.proxy_model.itemData(self.proxy_model.index(index.row(),1))
	# 			co.fxn()
	# 			self.refocus_le()

	def refocus_le(self):
		self.le_filter.clear()
		self.proxy_view.clearSelection()
		self.le_filter.setFocus()

	def update_commands(self):
		self.model.blockSignals(True)
		for k,v in list(self.commands.items()):
			x = self.model.findItems(k)
			if len(x) == 0:
				self.model.insertRow(0)
				self.model.setData(self.model.index(0,0),k)
				self.model.item(0,0).setEditable(False)
				co = command_obj(k,v)
				self.model.setData(self.model.index(0,1),co)
				# self.model.item(0,1).setIcon(self.style().standardIcon(getattr(QStyle,'SP_MediaPlay')))
				# self.model.item(0,1).setEditable(False)
		self.proxy_model.setSourceModel(self.model)
		self.proxy_view.sortByColumn(0, Qt.AscendingOrder)
		self.model.blockSignals(False)
		self.resize_columns()

	def keyPressEvent(self,event):
		if event.key() == Qt.Key_Escape:
			if self.le_filter.hasFocus() and not str(self.le_filter.text()) is "":
				self.le_filter.clear()
				return
			self.le_filter.setFocus()
			self.proxy_view.clearSelection()
			super(preferences,self).keyPressEvent(event)

		elif event.key() == Qt.Key_Return and self.le_filter.hasFocus():
			clicksign = 'Execute'
			if self.proxy_model.rowCount() > 0:
				# if self.proxy_model.itemData(self.proxy_model.index(0,0))[0] == self.le_filter.text():
					# self.proxy_model.itemData(self.proxy_model.index(0,1))[0].fxn()
					# self.le_filter.clear()
					# self.proxy_view.clearSelection()
					# self.le_filter.setFocus()
				# else:
				self.focusNextChild()
				self.proxy_view.setCurrentIndex(self.proxy_model.index(0,1))
				return

	def edit(self,a):
		name = self.model.data(self.model.index(a.row(),0))
		new_value = self.model.data(self.model.index(a.row(),1))
		old_value = self.model.old_value

		## Customs
		if name == 'plot_colormap':
			if new_value not in cm.__dict__:
				self.model.blockSignals(True)
				self['plot_colormap'] = 'viridis'
				self.model.blockSignals(False)
			self.redraw()
		self.log(name,old_value,new_value)

		if name == 'pref_precision':
			if new_value > 0:
				self.delegate.precision = new_value

		self.edit_callback()

	def filter_regex_changed(self):
		syntax = QRegExp.PatternSyntax(QRegExp.RegExp)
		# QRegExp.RegExp
		# QRegExp.Wildcard
		# QRegExp.FixedString
		regExp = QRegExp(self.le_filter.text(), Qt.CaseInsensitive, syntax)
		self.proxy_model.setFilterRegExp(regExp)

	def __setitem__(self, key, val):
		### OVERRIDE TO MAKE IT ACT LIKE A DICTIONARY
		## window['s'] = 'asdf'
		self.add_dictionary({key:val})

	def __getitem__(self, key):
		### OVERRIDE TO MAKE IT ACT LIKE A DICTIONARY
		## 	print(window['s'])
		val = self.get(key)
		return val

	def count(self,k):
		x = self.model.findItems(k)
		return len(x)

	def add_dictionary(self,dictionary):
		self.model.blockSignals(True)
		for k,v in list(dictionary.items()):
			if type(v) is list or type(v) is tuple:
				v = str(v)
			x = self.model.findItems(k)
			if len(x) > 0:
				self.model.setData(self.model.index(x[0].row(),1),v)
			else:
				self.model.insertRow(0)
				self.model.setData(self.model.index(0,0),k)
				self.model.item(0,0).setEditable(False)
				self.model.setData(self.model.index(0,1),v)
		self.proxy_model.setSourceModel(self.model)
		self.proxy_view.sortByColumn(0, Qt.AscendingOrder)
		self.model.blockSignals(False)
		self.resize_columns()

	def resize_columns(self):
		w = self.proxy_view.size().width()
		self.proxy_view.setColumnWidth(0,w/2)

	def get(self,s):
		x = self.model.findItems(s)
		if len(x) == 1:
			return self.model.ddata(self.model.index(x[0].row(),1))
		elif len(x) > 1:
			raise Exception('too many items named %s'%(s))
		raise Exception('no item %s'%(s))

	def undo(self):
		try:
			self.model.setData(self.model.old_index,self.model.old_value)
		except:
			pass

	### override these to put focus in line edit filter
	def setVisible(self,f):
		self.le_filter.setFocus()
		self.proxy_view.resizeColumnToContents(0)
		super(preferences,self).setVisible(f)

	def show(self):
		self.le_filter.setFocus()
		super(preferences,self).show()

	def output_str(self):
		total = ""
		for i in range(self.model.rowCount()):
			key = self.model.data(self.model.index(i,0))
			val = self.model.data(self.model.index(i,1))
			if type(val) is float:
				val = "{0:.{1}f}".format(val,self['pref_precision'])
			elif type(val) is type('a'):
				val = "\"%s\""%(str(val))
			if not key in self.commands:
				total += "%s:%s\n"%(key,val)
		return str(total)

	def load_str(self,s):
		d = {}
		ss =[ss.split(":") for ss in s.split('\n')]
		for sss in ss:
			try:
				if len(sss) == 2:
					k,v = sss
					if v[0] in ["\"","\'"]:
						v = v[1:-1]
					elif v in ['True','true','T','t']:
						v = True
					elif v in ['False','false','F','f']:
						v = False
					elif v.count('.') or v.count('e'):
						v = float(v)
					else: ## Will probably break if not anything yet and not int
						v = int(v)
					d[k] = v
			except:
				pass
		self.add_dictionary(d)

	def show(self):
		super(preferences,self).show()
		self.resize_columns()

	def load_preferences(self,fname=None):
		if fname is None:
			from PyQt5.QtCore import QFileInfo
			import os
			newfname = self.gui.latest_directory + os.sep + 'prefs.txt'
			fname = QFileDialog.getOpenFileName(self,'Choose preferences file to load',newfname)
			if fname[0] != "" and fname[1] != '':
				self.gui.latest_directory = QFileInfo(fname[0]).path()

			fname = fname[0]
		if fname != "":
			try:
				f = open(str(fname),'r')
				p = ""
				for line in f: p+=line
				f.close()
				self.load_str(p)
				try:
					self.gui.log('Loaded preferences from %s'%(fname),True)
				except:
					pass
				self.edit_callback()
			except:

				try:
					self.gui.log('Failed to load preferences from %s'%(fname),True)
				except:
					pass

	def save_preferences(self,fname=None):
		if fname is None:
			try:
				import os
				from PyQt5.QtCore import QFileInfo
				path = self.gui.latest_directory + os.sep
				newfilename = path + 'prefs.txt'
				fname = QFileDialog.getSaveFileName(self.gui, 'Save preferences text file', newfilename,'*.txt')
				if fname[0] != "" and fname[1] != '':
					self.gui.latest_directory = QFileInfo(fname[0]).path()
			except:
				fname = QFileDialog.getSaveFileName(self.gui, 'Save preferences text file', './prefs.txt','*.txt')

			fname = fname[0]

		if fname != "":
			try:
				p = self.output_str()
				f = open(fname,'w')
				f.write(p)
				f.close()
				try:
					self.gui.log('Saved preferences to %s'%(fname),True)
				except:
					pass
			except:
				try:
					self.gui.log('Failed to save preferences to %s'%(fname),True)
				except:
					pass


	####### Customs
	def redraw(self):
		try:
			self.gui.plot.image.set_cmap(self['plot_colormap'])
			self.gui.plot.canvas.draw()
		except:
			pass

	def log(self,name,old,new):
		try:
			self.gui.log('Pref: %s - %s > %s'%(name,old,new),True)
		except:
			pass


if __name__ == '__main__':

	import sys
	app = QApplication(sys.argv)
	prefs = preferences()
	q = {'fun list':[1,3,4],'aliens?':True,'plot_colormap':'jet'}
	prefs.add_dictionary(q)
	print(prefs['aliens?'])
	prefs['aliens?'] = False
	print(prefs['aliens?'])
	print(prefs['fun list'].mean())
	o = prefs.output_str()
	# oo = "fun list:\"yep\""

	# prefs.load_str(oo)
	prefs.show()
	sys.exit(app.exec_())
