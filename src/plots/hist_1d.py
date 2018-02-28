def hist_1d(gui):
	pass
#
# 	## Plot the 1D Histogram of the ensemble
# 	def hist1d(self):
# 		if self.docks['plots_1D'][0].isHidden():
# 			self.docks['plots_1D'][0].show()
# 		self.docks['plots_1D'][0].raise_()
# 		popplot = self.docks['plots_1D'][1]
# 		popplot.ax.cla()
#
# 		if self.ncolors == 2:
# 			fpb = self.get_plot_data()[0]
#
# 			# plt.hist(f.flatten(),bins=181,range=(-.4,1.4),histtype='stepfilled',alpha=.8,normed=True)
# 			popplot.ax.hist(fpb.flatten(),bins=self.gui.prefs['plotter_nbins_fret'],range=(self.gui.prefs['plotter_min_fret'],self.gui.prefs['plotter_max_fret']),histtype='stepfilled',alpha=.8,normed=True)
#
# 			if not self.hmm_result is None:
# 				r = self.hmm_result
# 				def norm(x,m,v):
# 					return 1./np.sqrt(2.*np.pi*v)*np.exp(-.5/v*(x-m)**2.)
# 				x = np.linspace(self.gui.prefs['plotter_min_fret'],self.gui.prefs['plotter_max_fret'],1001)
# 				ppi = np.sum([r.gamma[i].sum(0) for i in range(len(r.gamma))],axis=0)
# 				ppi /=ppi.sum()
# 				v = r.b/r.a
# 				tot = np.zeros_like(x)
# 				for i in range(r.m.size):
# 					y = ppi[i]*norm(x,r.m[i],v[i])
# 					tot += y
# 					popplot.ax.plot(x,y,color='k',lw=1,alpha=.8,ls='--')
# 				popplot.ax.plot(x,tot,color='k',lw=2,alpha=.8)
#
# 			popplot.ax.set_xlim(self.gui.prefs['plotter_min_fret'],self.gui.prefs['plotter_max_fret'])
# 			popplot.ax.set_xlabel(r'$\rm E_{\rm FRET}(t)$',fontsize=14./self.canvas.devicePixelRatio())
# 			popplot.ax.set_ylabel('Probability',fontsize=14./self.canvas.devicePixelRatio())
# 			for asp in ['top','bottom','left','right']:
# 				popplot.ax.spines[asp].set_linewidth(1.0/self.canvas.devicePixelRatio())
# 			popplot.f.subplots_adjust(left=.13,bottom=.15,top=.95,right=.99)
# 			popplot.f.canvas.draw()
