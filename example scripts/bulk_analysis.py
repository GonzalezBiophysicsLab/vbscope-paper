import os

import sys
sys.path.insert(0,'/home/colin/programs/python_scripts/vbscope')
from src.ui import launch_vbscope, launch_plotter

class experiment(object):
	def __init__(self,dir,start_filter=None,exclude_filter=None,filetype='.tif'):
		self.dir = dir
		self.filetype = filetype
		self.start_filter = start_filter
		self.exclude_filter = exclude_filter
		self.get_files()

	def get_files(self):
		self.fnames = []
		self.files_only = []
		if not self.dir is None:
			for root, dirs, files in os.walk(self.dir):
				for filename in files:
					if filename.endswith(self.filetype):
						if self.start_filter is None:
							self.fnames.append(os.path.join(root,filename))
							self.files_only.append(filename)
						else:
							if filename.startswith(self.start_filter):
								if self.exclude_filter is None:
									self.fnames.append(os.path.join(root,filename))
									self.files_only.append(filename)
								else:
									if filename.count(self.exclude_filter) == 0:
										self.fnames.append(os.path.join(root,filename))
										self.files_only.append(filename)

experiments = [
	experiment('/media/colin/Seagate Backup Plus Drive/colin/microscope/20180418 vc436H5S',start_filter='vc436_H5S',exclude_filter='highpower',filetype='.tif')#,
	# experiment('/media/colin/Seagate Backup Plus Drive/colin/microscope/20180418 vc436H5S',start_filter='vc436_H5S_highpower',filetype='.tif')
]
align_filename = 'vbscope_alignment_888_2x_20180316.dat'
align_filename4 = 'vbscope_alignment_888_4x_20180316.dat'
align_filename8 = 'vbscope_alignment_888_8x_20180316.dat'

gui = launch_vbscope(True)
gui.closeEvent = gui.unsafe_close

# gui.docks['transform'][1].load(fname=align_filename)
# print 1
print experiments[0].fnames
for e in experiments:
	for i in range(len(e.fnames)):

		fname = e.fnames[i]
		gui.load(fname=fname)
		if gui.data.movie.shape[1] == 256:
			gui.docks['transform'][1].load(fname=align_filename4)
			gui.prefs['extract_binning'] = 4
			gui.prefs['extract_nintegrate'] = 5
		elif gui.data.movie.shape[1] == 128:
			gui.docks['transform'][1].load(fname=align_filename8)
			gui.prefs['extract_binning'] = 8
			gui.prefs['extract_nintegrate'] = 3
		else:
			gui.docks['transform'][1].load(fname=align_filename)
			gui.prefs['extract_binning'] = 2
			gui.prefs['extract_nintegrate'] = 7

		# print 'loaded'
		gui.docks['background'][1].combo_method.setCurrentIndex(2) # median
		gui.docks['background'][1].update_background()

		# print 'background'

		gui.docks['spotfind'][1].bb = 1.00 # p=1.0
		gui.docks['spotfind'][1].pp = 0.95 # p = 0.90
		if gui.data.movie.shape[1] == 512:
			gui.docks['spotfind'][1].spin_start.setValue(7)
			gui.docks['spotfind'][1].spin_end.setValue(47) ## first second
		elif gui.data.movie.shape[1] == 256:
			gui.docks['spotfind'][1].spin_start.setValue(14)
			gui.docks['spotfind'][1].spin_end.setValue(100) ## first second
		elif gui.data.movie.shape[1] == 128:
			gui.docks['spotfind'][1].spin_start.setValue(30)
			gui.docks['spotfind'][1].spin_end.setValue(200) ## first second
		gui.docks['spotfind'][1].findspots()

		# gui.docks['transform'][1].stochastic()
		print 'spotfind %s'%(fname)
		gui.docks['extract'][1].extract()
		gui.docks['extract'][1].launch_plotter()
		if gui.docks['extract'][1].ui_p.data.d.shape[0] > 0:
			try:
				gui.docks['extract'][1].ui_p.closeEvent = gui.docks['extract'][1].ui_p.unsafe_close
				gui.docks['extract'][1].ui_p.data.cull_min(threshold=-10000)
				gui.docks['extract'][1].ui_p.data.cull_max(threshold=65535)
				# gui.docks['extract'][1].ui_p.data.remove_dead(threshold=10.)
				gui.docks['extract'][1].ui_p.data.cull_photons(color='0',threshold=0)
				gui.docks['extract'][1].ui_p.data.cull_photons(color='0+1',threshold=0)
				fname
				gui.docks['extract'][1].ui_p.export_traces(oname=fname[:-4]+'_traces.dat')
			except:
				print 'error'
		gui.docks['extract'][1].ui_p.close()


f = open('log.txt','w')
f.write(gui._log.textedit.toPlainText())
f.close()

gui.close()
