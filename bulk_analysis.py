from src.ui import launch_vbscope

gui = launch_vbscope(True)
gui.closeEvent = gui.unsafe_close

gui.docks['transform'][1].load('alignment_file.dat')

fnames = ['5.tif','5.tif']

for i in range(len(fnames)):
	fname = fnames[i]
	gui.load(fname)

	gui.docks['background'][1].combo_method.setCurrentIndex(4)

	gui.docks['spotfind'][1].sliders[0].setValue(1000) # p=1.0
	gui.docks['spotfind'][1].spin_start.setValue(9)
	gui.docks['spotfind'][1].spin_end.setValue(200)
	gui.docks['spotfind'][1].findspots()

	# gui.docks['transform'][1].stochastic()

	gui.docks['extract'][1].extract()
	gui.docks['extract'][1].launch_plotter()
	gui.docks['extract'][1].ui_p.closeEvent = gui.docks['extract'][1].ui_p.unsafe_close
	gui.docks['extract'][1].ui_p.data.remove_dead(10.)
	gui.docks['extract'][1].ui_p.export_traces('traces_%s.dat'%(fname))
	gui.docks['extract'][1].ui_p.close()
	
gui.close()
