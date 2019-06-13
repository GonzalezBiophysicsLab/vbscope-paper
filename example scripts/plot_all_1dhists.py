import os, sys
# sys.path.insert(0,'/home/colin/programs/python_scripts/vbscope')
from smfret_plotter.ui import launch_vbscope, launch_plotter
from smfret_plotter.plots import hist_1d

# prefixes = [
# 	'./20180321 vc436LX/vc436LX',
# 	'./20180321 vc436HX/vc436HX',
# 	# './20180321 vc211HS/vc211HS',
# 	'../20180320 vc436/20180320 vc436LS/vc436LS',
# 	'../20180320 vc436/20180320 vc436HS/vc436HS',
# 	'../20180418 vc436H5S/vc436_H5S'
# ]
prefixes = ['./']

for dir in ['./histograms/nstates','./histograms/1D_raw','./histograms/1D_viterbi']:
	try:
		os.makedirs(dir)
	except:
		pass

gui = launch_plotter(True)
gui.closeEvent = gui.unsafe_close

for pre in prefixes:
	## 1. Load Data
	gui.load_traces(filename=pre+'_traces.dat')
	gui.load_classes(filename=pre+'_classes.dat')

	gui.plot.initialize_plots()

	## 2. Select Classes
	for m in gui.action_classes:
		m.setChecked(False)

	gui.action_classes[0].setChecked(True)
	# gui.action_classes[1].setChecked(True)
	# gui.action_classes[2].setChecked(True)

	## 3. Remove Acceptor Photobleaching
	gui.data.remove_acceptor_bleach_from_fret()

	## 4. Set vbFRET priors
	gui.prefs['prior_a'] = 1.0
	gui.prefs['prior_beta'] = 0.1


	## 5. Run vbFRET + Model Selection
	gui.data.run_vbhmm_model(1,10,'E_FRET',False)


	## testing
	pre = pre.split('/')[-1]
	print

	## 6. Plot N states histogram
	gui.plot_vb_states()
	plot = gui.popout_plots['vb_states'].ui
	plot.canvas.figure.savefig('./histograms/nstates/nstates_histogram_'+pre+'_c1.pdf')

	## 7. Plot 1D Histogram w/ Posterior Predictive
	gui.plot_hist1d()
	plot = gui.popout_plots['plot_hist1d'].ui
	plot.prefs['fig_height'] = 4.0
	plot.prefs['hist_force_ymax'] = True
	plot.prefs['hist_ymax'] = 2.5
	plot.prefs['label_y_nticks'] = 6
	plot.prefs['textbox_x'] = 0.96
	plot.prefs['textbox_fontsize'] = 10.0
	hist_1d.recalc(gui)
	plot.canvas.figure.savefig('./histograms/1D_raw/1D_raw_histogram_'+pre+'_c1.pdf')

	## 8. Plot KDE of the Viterbi paths
	plot.prefs['hmm_states'] = True
	plot.prefs['hmm_viterbi'] = True
	plot.prefs['hist_on'] = False
	hist_1d.recalc(gui)
	plot.canvas.figure.savefig('./histograms/1D_viterbi/1D_viterbi_histogram_'+pre+'_c1.pdf')


	# ## 1
	# gui.action_classes[2].setChecked(False)
	# # plot.prefs['filter']=False
	# plot.replot()
	# plot.canvas.figure.savefig('./hists/hist1d_'+pre[-7:]+'_c1.pdf')
	# plot.prefs['filter']=True
	# plot.replot()
	# plot.canvas.figure.savefig('./hists/hist1d_'+pre[-7:]+'_c1_filter.pdf')
	#
	# ## 1+2
	# gui.action_classes[2].setChecked(True)
	# plot.prefs['filter']=False
	# plot.replot()
	# plot.canvas.figure.savefig('./hists/hist1d_'+pre[-7:]+'_c12.pdf')
	# plot.prefs['filter']=True
	# plot.replot()
	# plot.canvas.figure.savefig('./hists/hist1d_'+pre[-7:]+'_c12_filter.pdf')
sys.exit(gui.app.exec_())
