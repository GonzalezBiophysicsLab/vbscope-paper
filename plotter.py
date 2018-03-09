from src.ui import launch_plotter

gui = launch_plotter(False)


# # sample script
# gui = launch_plotter(True)
# gui.load_traces(filename='_traces.dat')
# gui.load_classes(filename='_classes.dat')
# gui.plot_hist1d()
# popplot = gui.popout_plots['plot_hist1d'].ui
# popplot.prefs['plotter_min_fret']=-.25
# popplot.prefs['plotter_max_fret']=1.25
# gui.plot_hist1d()
# popplot.f.savefig('test.pdf')
