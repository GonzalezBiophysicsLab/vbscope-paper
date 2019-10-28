import vbscope

fdir = '/Volumes/crypt/smfret_data/'
fname = 'vc436hxlx+vc211hs.hdf5'
alignment_name = 'vbscope_alignment_888_2x_20180316.dat'
prefixes = ['vc211HS','vc436HX','vc436LX']

import h5py as h
with h.File(fdir+fname,'r') as f:
	dnames = [k for k in f.keys()]


def subset_process(theta):
	prefix,fdir,fname,alignment_name,dnames = theta

	gui = vbscope.launch_scriptable()
	gui.docks['spotfind'][1].spin_start.setValue(7)
	gui.prefs['extract_same_cutoff'] = 2.0
	for i in range(len(dnames)):
		dname = dnames[i]
		if dname.startswith(prefix):
			gui._load_hdf5_dataset(fdir+fname,dnames[i])
			gui.docks['spotfind'][1].threshold_find()
			s = [gui.docks['spotfind'][1].xys[j].shape[-1] for j in range(gui.data.ncolors)]
			print(dname,s)
			gui.docks['transform'][1].load(fname=fdir+alignment_name)
			# gui.docks['extract'][1].combo_method.setCurrentIndex(0) ## center
			gui.docks['extract'][1].combo_method.setCurrentIndex(1) ## mlpsf
			gui.docks['extract'][1].extract()
			gui.docks['extract'][1].launch_plotter()
			gui.docks['extract'][1].ui_p.export_hdf5(fdir+'traces/' + dname[:-21]+'.hdf5')
			print(dname,gui.docks['extract'][1].ui_p.data.raw.shape)
			gui.docks['extract'][1].ui_p.quick_close()

	gui.quick_close()


from multiprocessing import Pool
with Pool(processes=len(prefixes)) as pool:
	pool.map(subset_process, [[prefix,fdir,fname,alignment_name,dnames] for prefix in prefixes])
