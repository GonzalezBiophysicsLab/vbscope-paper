from smfret_plotter import launch

gui = launch(False)

# # sample script
# gui = launch_plotter(True)
# gui.load_traces(filename='_traces.dat')
# gui.load_classes(filename='_classes.dat')
# gui.plot_hist1d()
# popplot = gui.popout_plots['plot_hist1d'].ui
# popplot.prefs['plot_fret_min']=-.25
# popplot.prefs['plot_fret_max']=1.25
# gui.plot_hist1d()
