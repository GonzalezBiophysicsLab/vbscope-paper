import numpy as np
from .ui_popplot import popplot_widget

class psf_tool(popplot_widget):
	def __init__(self,gui=None):
		super(psf_tool,self).__init__(nax1=gui.data.ncolors,nax2=4)
		self.gui = gui
		self.ncolors = self.gui.data.ncolors
		self.add_menu_function('Pull Spots',self.pull_spots)
		# self.colors = self.add_menu_group('Color Channels',["Color %d"%(1+i) for i in range(self.ncolors)])
		self.canvas.draw()

	def pull_spots(self):
		spots = self.gui.docks['spotfind'][1].xys
		if spots is None:
			return

		from scipy.optimize import minimize
		from math import erf as erff
		import numba as nb
		@nb.vectorize
		def erf(x):
			return erff(x)

		def psf_fxn(gx,gy,mx,my,a,b,sx,sy):
			taux = 1./sx/sx
			tauy = 1./sy/sy
			ex = .5 * (erf(taux*(gx+.5-mx)) - erf(taux*(gx-.5-mx)))
			ey = .5 * (erf(tauy*(gy+.5-my)) - erf(tauy*(gy-.5-my)))
			z0 = 2.#.25 * (erf(tau*(.5-mx)) - erf(tau*(-.5-mx)))*(erf(tau*(.5-my)) - erf(tau*(-.5-my)))
			## z0 sets (0,0) to 1
			z = a*ex*ey/z0 + b
			return z

		def fitfxn(theta,z,gx,gy):
			return np.sum((z-psf_fxn(gx,gy,*theta))**2.)

		img = self.gui.data.movie.max(0)

		for color in range(self.ncolors):
			mask = np.zeros_like(img).astype('int32')
			cspots = spots[color]
			for i in range(len(cspots[0])):
				mask[cspots[0][i],cspots[1][i]] = 1

			l = (self.gui.prefs['extract_nintegrate']-1)//2
			punches = punch_out(img,mask,l=l)
			cmap = self.gui.prefs['plot_colormap']
			self.ax[color][0].imshow(mask,cmap=cmap)
			## 1
			punches /= punches.sum((1,2))[:,None,None]
			psf = np.median(punches,axis=0)
			## 2
			# psf = punches.mean(0)
			## 3
			# punches -= punches.min((1,2))[:,None,None]
			# punches /= punches.sum((1,2))[:,None,None]
			psf = np.mean(punches,axis=0)
			psf -= psf.min()
			psf /= psf.max()
			self.psf = psf

			xy = -np.arange(-l,l+1).astype('float64')
			gx,gy = np.meshgrid(xy,xy)
			guess = np.array((0.,0.,psf.max(),0.,1.,1.))
			out = minimize(fitfxn,x0=guess,args=(self.psf,gx,gy),method='Nelder-Mead',options={'maxiter':10000})
			model = psf_fxn(gx,gy,*out.x)
			# model = psf_fxn(gx,gy,*guess)

			cmax = self.psf.max()
			cmin = self.psf.min()
			delta = cmax-cmin
			cmax += 0.05*delta
			cmin -= 0.05*delta
			cl = (cmax-cmin)/2.
			self.ax[color][1].imshow(self.psf,cmap=cmap,vmin=cmin,vmax=cmax)
			# x,y = radial_profile(psf)
			# self.ax[2].plot(x,y)
			self.ax[color][2].imshow(model,cmap=cmap,vmin=cmin,vmax=cmax)
			self.ax[color][3].imshow(self.psf-model,cmap=cmap,vmin=-cl,vmax=cl)

			for aa in self.ax[color]:
				aa.set_xticks((),())
				aa.set_yticks((),())
			self.ax[color][0].set_ylabel('Color %d'%(color+1))
			self.ax[color][1].set_title("sx:%.2f, sy:%.2f"%(np.abs(out.x[-2]),np.abs(out.x[-1])))
		self.fig.subplots_adjust(hspace=0.05,wspace=.02,left=.1,right=.98,top=.95,bottom=.02)
		self.canvas.draw()
			# self.gui.data.movie

	# def get_sigma(self,j):
	# 	''' j is the color index to pick the wavelenght of light '''
	#
	# 	c = 0.42 # .45 or .42, airy disk to gaussian
	# 	psf_sig = c*self.gui.prefs['channels_wavelengths'][j]*self.gui.prefs['extract_numerical_aperture']
	# 	sigma = psf_sig/self.gui.prefs['extract_pixel_size']*self.gui.prefs['extract_magnification']/self.gui.prefs['extract_binning']
	# 	return sigma


def punch_out(movie,mask,l=11,value=0.):
	out = np.zeros((mask.sum(),2*l+1,2*l+1)) + value
	y,z = np.nonzero(mask)
	for i in range(out.shape[0]):
		my = y[i]
		mz = z[i]
		ymin = max(0,my-l)
		ymax = min(movie.shape[0],my+l+1)
		zmin = max(0,mz-l)
		zmax = min(movie.shape[1],mz+l+1)

		out[i,l-(my-ymin):l+(ymax-my),l-(mz-zmin):l+(zmax-mz)] = movie[ymin:ymax,zmin:zmax]
	return out



if __name__ == '__main__':

	## View data
	import sys
	app = QApplication(sys.argv)
	view = psf_tool()
	view.show()

	sys.exit(app.exec_())
