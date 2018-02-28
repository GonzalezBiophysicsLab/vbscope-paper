def tdp(gui):
	pass

#
# 	def tdplot(self):
# 		if self.docks['plots_TDP'][0].isHidden():
# 			self.docks['plots_TDP'][0].show()
# 		self.docks['plots_TDP'][0].raise_()
# 		popplot = self.docks['plots_TDP'][1]
# 		popplot.clf()
#
# 		if self.ncolors == 2:
# 			fpb = self.get_plot_data()[0]
# 			d = np.array([[fpb[i,:-1],fpb[i,1:]] for i in range(fpb.shape[0])])
#
# 			if not self.hmm_result is None:
# 				# state,success = QInputDialog.getInt(self,"Pick State","Which State?",min=0,max=self.hmm_result.nstates-1)
# 				# if success:
# 				v = self.get_viterbi_data()
# 				vv = np.array([[v[i,:-1],v[i,1:]] for i in range(v.shape[0])])
#
# 				for i in range(d.shape[0]):
# 					d[i,:,vv[i,0]==vv[i,1]] = np.array((np.nan,np.nan))
#
# 			rx = np.linspace(self.gui.prefs['plotter_min_fret'],self.gui.prefs['plotter_max_fret'],self.gui.prefs['plotter_nbins_fret'])
# 			ry = np.linspace(self.gui.prefs['plotter_min_fret'],self.gui.prefs['plotter_max_fret'],self.gui.prefs['plotter_nbins_fret'])
# 			x,y = np.meshgrid(rx,ry,indexing='ij')
# 			dx = d[:,0].flatten()
# 			dy = d[:,1].flatten()
# 			cut = np.isfinite(dx)*np.isfinite(dy)
# 			z,hx,hy = np.histogram2d(dx[cut],dy[cut],bins=[rx.size,ry.size],range=[[rx.min(),rx.max()],[ry.min(),ry.max()]])
#
# 			from scipy.ndimage import gaussian_filter
# 			z = gaussian_filter(z,(self.gui.prefs['plotter_smoothx'],self.gui.prefs['plotter_smoothy']))
#
# 			try:
# 				cm = plt.cm.__dict__[self.gui.prefs['plotter_cmap']]
# 			except:
# 				cm = plt.cm.rainbow
# 			try:
# 				cm.set_under(self.gui.prefs['plotter_floorcolor'])
# 			except:
# 				cm.set_under('w')
#
# 			from matplotlib.colors import LogNorm
# 			if self.gui.prefs['plotter_floor'] <= 0:
# 				bins = np.logspace(np.log10(z[z>0.].min()),np.log10(z.max()),self.gui.prefs['plotter_nbins_contour'])
# 				pc = popplot.ax.contourf(x, y, z, bins, cmap=cm, norm=LogNorm())
# 			else:
# 				z[z< 1e-10] = 1e-9
# 				bins = np.logspace(0,np.log10(z.max()),self.gui.prefs['plotter_nbins_contour'])
# 				bins = np.append(1e-10,bins)
# 				pc = popplot.ax.contourf(x, y, z, bins, vmin=self.gui.prefs['plotter_floor'], cmap=cm, norm=LogNorm())
#
# 			for pcc in pc.collections:
# 				pcc.set_edgecolor("face")
#
# 			cb = popplot.f.colorbar(pc)
# 			zm = np.floor(np.log10(z.max()))
# 			cz = np.logspace(0,zm,zm+1)
# 			cb.set_ticks(cz)
# 			# cb.set_ticklabels(cz)
# 			cb.ax.yaxis.set_tick_params(labelsize=12./self.canvas.devicePixelRatio(),direction='in',width=1.0/self.canvas.devicePixelRatio(),length=4./self.canvas.devicePixelRatio())
# 			for asp in ['top','bottom','left','right']:
# 				cb.ax.spines[asp].set_linewidth(1.0/self.canvas.devicePixelRatio())
# 			cb.solids.set_edgecolor('face')
# 			cb.solids.set_rasterized(True)
#
# 			popplot.ax.set_xlim(rx.min(),rx.max())
# 			popplot.ax.set_ylim(ry.min(),ry.max())
# 			popplot.ax.set_xlabel(r'Initial E$_{\rm FRET}$',fontsize=14./self.canvas.devicePixelRatio())
# 			popplot.ax.set_ylabel(r'Final E$_{\rm FRET}$',fontsize=14./self.canvas.devicePixelRatio())
# 			popplot.ax.set_title('Transition Density (Counts)',fontsize=12/self.canvas.devicePixelRatio())
# 			bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=1./self.canvas.devicePixelRatio())
# 			popplot.ax.annotate('n = %d'%(fpb.shape[0]),xy=(.95,.93),xycoords='axes fraction',ha='right',color='k',bbox=bbox_props,fontsize=12./self.canvas.devicePixelRatio())
#
# 			for asp in ['top','bottom','left','right']:
# 				popplot.ax.spines[asp].set_linewidth(1.0/self.canvas.devicePixelRatio())
# 			popplot.f.subplots_adjust(left=.18,bottom=.14,top=.92,right=.99)
#
# 			popplot.f.canvas.draw()
