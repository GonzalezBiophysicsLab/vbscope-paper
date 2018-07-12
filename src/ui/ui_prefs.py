from PyQt5.QtWidgets import QMainWindow,QWidget,QHBoxLayout,QSizePolicy,QLineEdit, QTableView, QVBoxLayout, QWidget, QApplication, QStyledItemDelegate, QDoubleSpinBox, QShortcut
from PyQt5.QtCore import  QRegExp, QSortFilterProxyModel, Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QKeySequence

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
	'ui_version':0.1,
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
				return old
		return new

	def ddata(self,*args):
		d = super(QAlmostStandardItemModel,self).data(*args)
		if type(d) is type(u'd'):
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

	def createEditor(self, parent, option, index):
		value = index.data()
		editor = super(precision_delegate,self).createEditor(parent,option,index)
		if isinstance(value, float):
			editor.setDecimals(self.precision)
		return editor

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
		self.proxy_view.setItemDelegate(self.delegate)

		self.proxy_view.setModel(self.proxy_model)
		self.proxy_view.setSortingEnabled(True)
		self.proxy_view.horizontalHeader().setStretchLastSection(True)
		self.proxy_view.horizontalHeader().setVisible(False)
		self.proxy_view.verticalHeader().setVisible(False)

		self.le_filter = QAlmostLineEdit()
		self.le_filter.textChanged.connect(self.filter_regex_changed)
		self.le_filter.setPlaceholderText("Enter filter here")

		proxyLayout = QVBoxLayout()
		proxyLayout.addWidget(self.le_filter)
		proxyLayout.addWidget(self.proxy_view)
		self.setLayout(proxyLayout)

		self.setWindowTitle("Preferences")

		self.model.itemChanged.connect(self.edit)
		self.edit_callback = lambda: None

		self.add_dictionary(default_prefs)
		self.le_filter.setFocus()

		undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"),self)
		undo_shortcut.activated.connect(self.undo)

	def keyPressEvent(self,event):
		if event.key() == Qt.Key_Escape:
			if self.le_filter.hasFocus() and not str(self.le_filter.text()) is "":
				self.le_filter.clear()
				return
			self.le_filter.setFocus()
			self.proxy_view.clearSelection()
			super(preferences,self).keyPressEvent(event)
		elif event.key() == Qt.Key_Return and self.le_filter.hasFocus():
			if self.proxy_model.rowCount() > 0:
				self.focusNextChild()

	def edit(self,a):
		name = self.model.data(self.model.index(a.row(),0))
		new_value = self.model.data(self.model.index(a.row(),1))
		old_value = self.model.old_value

		## Customs
		if name == 'plot_colormap':
			if not cm.__dict__.has_key(new_value):
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
		## 	print window['s']
		val = self.get(key)
		return val

	def count(self,k):
		x = self.model.findItems(k)
		return len(x)

	def add_dictionary(self,dictionary):
		self.model.blockSignals(True)
		for k,v in dictionary.items():
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
		self.resize_columns()
		self.model.blockSignals(False)

	def resize_columns(self):
		self.proxy_view.setFocus()
		self.proxy_view.hide()
		self.proxy_view.resizeColumnsToContents()
		self.proxy_view.show()

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
		self.resize_columns()
		super(preferences,self).setVisible(f)

	def show(self):
		self.le_filter.setFocus()
		self.resize_columns()
		super(preferences,self).show()

	####### Customs
	def combine_prefs(self,add_dictionary):
		## legacy
		self.add_dictionary(add_dictionary)

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
	print prefs['aliens?']
	prefs['aliens?'] = False
	print prefs['aliens?']
	print prefs['fun list'].mean()
	prefs.show()
	sys.exit(app.exec_())
