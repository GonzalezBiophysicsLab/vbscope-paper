
#
# 	def hist2d(self):
# 		if self.docks['plots_2D'][0].isHidden():
# 			self.docks['plots_2D'][0].show()
# 		self.docks['plots_2D'][0].raise_()
# 		popplot = self.docks['plots_2D'][1]
# 		popplot.clf()
#
# 		if self.ncolors == 2:
# 			fpb = self.get_plot_data()[0]
# 			if self.gui.prefs['synchronize_start_flag'] == 'True':
# 				print np.nansum(fpb)
# 				for i in range(fpb.shape[0]):
# 					y = fpb[i].copy()
# 					fpb[i] = np.nan
# 					pre = self.pre_list[i]
# 					post = self.pb_list[i]
# 					if pre < post:
# 						fpb[i,0:post-pre] = y[pre:post]
# 				print np.nansum(fpb)
# 			elif not self.hmm_result is None:
# 				state,success = QInputDialog.getInt(self,"Pick State","Which State?",min=0,max=self.hmm_result.nstates-1)
# 				if success:
# 					v = self.get_viterbi_data()
# 					vv = np.array([[v[i,:-1],v[i,1:]] for i in range(v.shape[0])])
# 					oo = []
# 					for i in range(fpb.shape[0]):
# 						ms = np.nonzero((vv[i,1]==state)*(vv[i,0]!=vv[i,1]))[0]
# 						if v[i,0] == state:
# 							ms = np.append(0,ms)
# 						ms = np.append(ms,v.shape[1])
#
# 						for j in range(ms.size-1):
# 							o = fpb[i].copy()
# 							ox = int(np.max((0,ms[j]-self.gui.prefs['plotter_2d_syncpreframes'])))
# 							o = o[ox:ms[j+1]]
# 							ooo = np.empty(v.shape[1]) + np.nan
# 							ooo[:o.size] = o
# 							oo.append(ooo)
# 					fpb = np.array(oo)
#
#
# 			dtmin = self.gui.prefs['plotter_min_time']
# 			dtmax = self.gui.prefs['plotter_max_time']
# 			if dtmax == -1:
# 				dtmax = fpb.shape[1]
# 			dt = np.arange(dtmin,dtmax)*self.gui.prefs['tau']
# 			ts = np.array([dt for _ in range(fpb.shape[0])])
# 			fpb = fpb[:,dtmin:dtmax]
# 			xcut = np.isfinite(fpb)
# 			bt = (self.gui.prefs['plotter_max_fret'] - self.gui.prefs['plotter_min_fret']) / (self.gui.prefs['plotter_nbins_fret'] + 1)
# 			z,hx,hy = np.histogram2d(ts[xcut],fpb[xcut],bins = [self.gui.prefs['plotter_nbins_time'],self.gui.prefs['plotter_nbins_fret']+2],range=[[dt[0],dt[-1]],[self.gui.prefs['plotter_min_fret']-bt,self.gui.prefs['plotter_max_fret']+bt]])
# 			rx = hx[:-1]
# 			ry = .5*(hy[1:]+hy[:-1])
# 			x,y = np.meshgrid(rx,ry,indexing='ij')
#
# 			from scipy.ndimage import gaussian_filter
# 			z = gaussian_filter(z,(self.gui.prefs['plotter_smoothx'],self.gui.prefs['plotter_smoothy']))
#
# 			# cm = plt.cm.rainbow
# 			# vmin = self.gui.prefs['plotter_floor']
# 			# cm.set_under('w')
# 			# if vmin <= 1e-300:
# 			# 	vmin = z.min()
# 			# pc = plt.pcolor(y.T,x.T,z.T,cmap=cm,vmin=vmin,edgecolors='face')
# 			try:
# 				cm = plt.cm.__dict__[self.gui.prefs['plotter_cmap']]
# 			except:
# 				cm = plt.cm.rainbow
# 			try:
# 				cm.set_under(self.gui.prefs['plotter_floorcolor'])
# 			except:
# 				cm.set_under('w')
#
# 			vmin = self.gui.prefs['plotter_floor']
#
# 			if self.gui.prefs['plotter_2d_normalizecolumn'] == 'True':
# 				z /= np.nanmax(z,axis=1)[:,None]
# 			else:
# 				z /= np.nanmax(z)
#
# 			z = np.nan_to_num(z)
#
# 			x -= self.gui.prefs['plotter_timeshift']
#
# 			from matplotlib.colors import LogNorm
# 			if vmin <= 0 or vmin >=z.max():
# 				pc = popplot.ax.contourf(x.T,y.T,z.T,self.gui.prefs['plotter_nbins_contour'],cmap=cm)
# 			else:
# 				# pc = plt.pcolor(y.T,x.T,z.T,vmin =vmin,cmap=cm,edgecolors='face',lw=1,norm=LogNorm(z.min(),z.max()))
# 				pc = popplot.ax.contourf(x.T,y.T,z.T,self.gui.prefs['plotter_nbins_contour'],vmin =vmin,cmap=cm)
# 			for pcc in pc.collections:
# 				pcc.set_edgecolor("face")
#
# 			try:
# 				cb = popplot.f.colorbar(pc)
# 				cb.set_ticks(np.array((0.,.2,.4,.6,.8,1.)))
# 				cb.ax.yaxis.set_tick_params(labelsize=12./self.canvas.devicePixelRatio(),direction='in',width=1.0/self.canvas.devicePixelRatio(),length=4./self.canvas.devicePixelRatio())
# 				for asp in ['top','bottom','left','right']:
# 					cb.ax.spines[asp].set_linewidth(1.0/self.canvas.devicePixelRatio())
# 				cb.solids.set_edgecolor('face')
# 				cb.solids.set_rasterized(True)
# 			except:
# 				pass
#
# 			for asp in ['top','bottom','left','right']:
# 				popplot.ax.spines[asp].set_linewidth(1.0/self.canvas.devicePixelRatio())
# 			popplot.f.subplots_adjust(left=.18,bottom=.14,top=.95,right=.99)
#
# 			popplot.ax.set_xlim(rx.min()-self.gui.prefs['plotter_timeshift'],rx.max()-self.gui.prefs['plotter_timeshift'])
# 			popplot.ax.set_ylim(self.gui.prefs['plotter_min_fret'],self.gui.prefs['plotter_max_fret'])
# 			popplot.ax.set_xlabel('Time (s)',fontsize=14./self.canvas.devicePixelRatio())
# 			popplot.ax.set_ylabel(r'$\rm E_{\rm FRET}(t)$',fontsize=14./self.canvas.devicePixelRatio())
# 			bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=1./self.canvas.devicePixelRatio())
# 			popplot.ax.annotate('n = %d'%(fpb.shape[0]),xy=(.95,.93),xycoords='axes fraction',ha='right',color='k',bbox=bbox_props,fontsize=12./self.canvas.devicePixelRatio())
# 			popplot.canvas.draw()
