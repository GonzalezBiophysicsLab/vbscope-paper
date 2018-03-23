from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from matplotlib.widgets import  RectangleSelector


from PyQt5.QtWidgets import QSizePolicy,QVBoxLayout,QWidget

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import wiener

class traj_plot_container():
	'''
	.f - figure
	.ax - axis
	.toolbar - MPL toolbar
	'''
	def __init__(self,gui):
		self.gui = gui

		self.f,self.a = plt.subplots(2,2,gridspec_kw={'width_ratios':[6,1]},figsize=(6.5,4))
		self.canvas = FigureCanvas(self.f)
		self.f.set_dpi(self.f.get_dpi()/self.canvas.devicePixelRatio())

		sizePolicy = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
		self.canvas.setSizePolicy(sizePolicy)

		self.toolbar = NavigationToolbar(self.canvas,None)

		self.index = 0
		self.arm_blit()
		self.f.canvas.mpl_connect('resize_event',lambda e: self.arm_blit())

		self.canvas.draw()
		plt.close(self.f)

	def arm_blit(self):
		self.flag_arm = True

	def draw(self):
		self.canvas.update()
		self.canvas.flush_events()
		self.canvas.draw()

	## Read in y-min and y-max values, then update the plot
	def update_minmax(self):
		self.initialize_plots()
		self.yminmax = np.array((float(self.gui.le_min.text()),float(self.gui.le_max.text())))
		if not np.all(np.isfinite(self.yminmax)):
			self.yminmax = np.array((-1000.,10000))
			print "yminmax fixed"
		self.a[0][0].set_ylim(self.yminmax[0],self.yminmax[1])

		self.canvas.draw()

		for le in [self.gui.le_min,self.gui.le_max]:
			try:
				le.clearFocus()
			except:
				pass

	def plot_hist(self,i,hx,hy):
		self.a[0][1].lines[i].set_data(hy,hx)

	def plot_fret_hist(self,i,hx,hy):
		self.a[1][1].lines[i].set_data(hy,hx)

	def plot_no_hmm(self):
		self.a[1,0].lines[-1].set_data([0,0],[0,0])

	def plot_hmm(self,t,state_means,vitpath,pretime,pbtime):
		self.a[1,0].lines[-1].set_data(t[pretime:pbtime],state_means[vitpath])

	def plot_traj(self,t,intensities,rel,pretime,pbtime):
		for i in range(self.gui.ncolors):
			self.a[0][0].lines[3*i+0].set_data(t[:pretime],intensities[i,:pretime])
			self.a[0][0].lines[3*i+1].set_data(t[pretime:pbtime],intensities[i,pretime:pbtime])
			self.a[0][0].lines[3*i+2].set_data(t[pbtime:],intensities[i,pbtime:])

		for i in range(self.gui.ncolors-1):
			self.a[1][0].lines[3*i+0].set_data(t[:pretime],rel[i,:pretime])
			self.a[1][0].lines[3*i+1].set_data(t[pretime:pbtime],rel[i,pretime:pbtime])
			self.a[1][0].lines[3*i+2].set_data(t[pbtime:],rel[i,pbtime:])

	def calc_trajectory(self):
		intensities = self.gui.data.d[self.index].copy()

		if self.gui.prefs['convert_flag']:
			for i in range(self.gui.ncolors):
				intensities[i] = self.gui.prefs['convert_c_lambda'][i]/self.gui.prefs['convert_em_gain']*(intensities[i] - self.gui.prefs['convert_offset'])

		bts = self.gui.prefs['bleedthrough'].reshape((4,4))
		for i in range(self.gui.ncolors):
			for j in range(self.gui.ncolors):
				intensities[j] -= bts[i,j]*intensities[i]

		if self.gui.prefs['plotter_wiener_smooth'] != 0:
			ms = self.gui.prefs['plotter_wiener_smooth']
			if ms % 2 != 1:
				ms += 1 ## keep it odd
			for i in range(intensities.shape[0]):
				# from scipy.signal import savgol_filter
				# intensities[i] = savgol_filter(intensities[i],9,5)
				# intensities[i] = wiener(intensities[i])
				intensities[i] = wiener(intensities[i],mysize=ms)
				# # intensities = gaussian_filter1d(intensities,self.gui.prefs['plotter_smooth_sigma'],axis=1)

		t = np.arange(intensities.shape[1])*self.gui.prefs['tau']

		downsample = int(self.gui.prefs['downsample'])
		if downsample != 1:
			ll = t.size / downsample
			intensities = np.array([np.sum(intensities[i,:ll*downsample].reshape((ll,downsample)),axis=1) for i in range(self.gui.ncolors)])
			t = t[:ll*downsample].reshape((ll,downsample))[:,0]
		pbtime = int(self.gui.data.pb_list[self.index] / downsample)
		pretime = int(self.gui.data.pre_list[self.index] / downsample)

		rel = intensities[1:] / (1e-300+intensities.sum(0)[None,:])

		return t,intensities,rel,pretime,pbtime

	def calc_histograms(self,intensities,rel,pretime,pbtime):
		intensity_hists = []
		fret_hists = []
		hymaxes = []

		for i in range(self.gui.ncolors):
			if pretime < pbtime:
				hy,hx = np.histogram(intensities[i,pretime:pbtime],range=self.yminmax,bins=int(np.sqrt(pbtime-pretime)))
			else:
				hy = np.zeros(100)
				hx = np.linspace(self.yminmax[0],self.yminmax[1],101)
			hy = np.append(np.append(0.,hy),0.)
			hx = np.append(np.append(hx[0],.5*(hx[1:]+hx[:-1])),hx[-1])
			hymaxes.append(hy.max())
			intensity_hists.append([hx,hy])

		for i in range(self.gui.ncolors-1):
			if pretime < pbtime:
				hy,hx = np.histogram(rel[i,pretime:pbtime],range=(self.gui.prefs['plotter_min_fret'],self.gui.prefs['plotter_max_fret']),bins=int(np.sqrt(pbtime-pretime)))
			else:
				hy = np.zeros(100)
				hx = np.linspace(self.yminmax[0],self.yminmax[1],101)
			hy = np.append(np.append(0.,hy),0.)
			hx = np.append(np.append(hx[0],.5*(hx[1:]+hx[:-1])),hx[-1])
			hymaxes.append(hy.max())
			fret_hists.append([hx,hy])

		hymax = np.max(hymaxes)
		for i in range(len(intensity_hists)):
			intensity_hists[i][1] /= hymax
		for i in range(len(fret_hists)):
			fret_hists[i][1] /= hymax

		return intensity_hists,fret_hists

	def calc_hmm_traj(self):
		if self.gui.data.hmm_result.ran.count(self.index)>0:
			ii = self.gui.data.hmm_result.ran.index(self.index)
			vitpath = self.gui.data.hmm_result.viterbi[ii]
			if self.gui.prefs['hmm_binding_expt'] is True:
				state_means = np.array((0.,1.))
			else:
				state_means = self.gui.data.hmm_result.m
		else:
			state_means = None
			vitpath = None
		return state_means,vitpath

	def update_axes(self):
		self.a[0][0].set_xlim(0, self.gui.data.d.shape[2]*self.gui.prefs['tau'])
		self.a[0][0].set_ylim(self.yminmax[0],self.yminmax[1])
		self.a[0][1].set_xlim(0.01, 1.25)
		self.a[1][0].set_ylim(self.gui.prefs['plotter_min_fret'],self.gui.prefs['plotter_max_fret'])
		self.a[1][1].set_xlim(self.a[0][1].get_xlim())

	def update_colors(self):
		for i in range(self.gui.ncolors):
			color = self.gui.prefs['channel_colors'][i]
			for j in range(3):
				self.a[0][0].lines[3*i+j].set_color(color)
			self.a[0][1].lines[i].set_color(color)
		for i in range(self.gui.ncolors-1):
			if self.gui.ncolors == 2:
				color = 'blue'
			else:
				color = self.gui.prefs['channel_colors'][i+1]
			for j in range(3):
				self.a[1][0].lines[3*i+j].set_color(color)
			self.a[1][1].lines[i].set_color(color)

# 	## Plot current trajectory
	def update_plots(self):
		if self.flag_arm:
			self.update_blits()
			self.flag_arm = False
		[[self.f.canvas.restore_region(bbb) for bbb in bb] for bb in self.blit_bgs]

		t,intensities,rel,pretime,pbtime = self.calc_trajectory()
		self.plot_traj(t,intensities,rel,pretime,pbtime)

		if not self.gui.data.hmm_result is None:
			state_means,vitpath = self.calc_hmm_traj()
			if state_means is None:
				self.plot_no_hmm()
			else:
				self.plot_hmm(t,state_means,vitpath,pretime,pbtime)

		intensity_hists,fret_hists = self.calc_histograms(intensities,rel,pretime,pbtime)
		for i in range(len(intensity_hists)):
			self.plot_hist(i,*intensity_hists[i])
		for i in range(len(fret_hists)):
			self.plot_fret_hist(i,*fret_hists[i])
		self.update_colors()

		[[[aaa.draw_artist(l) for l in aaa.lines] for aaa in aa] for aa in self.a]
		[[self.f.canvas.blit(aaa.bbox) for aaa in aa] for aa in self.a]

		self.canvas.update()
		self.canvas.flush_events()

		if self.a[0][0].get_xlim()[1] != self.gui.data.d.shape[2]*self.gui.prefs['tau']:
			self.update_axes()
		self.draw()

	## Plot initial data to set aesthetics
	def initialize_plots(self):
		## clear everything
		[[aaa.cla() for aaa in aa] for aa in self.a]

		lw = .75 / self.canvas.devicePixelRatio()
		pb = .2

		## Make it so that certain plots zoom together
		self.a[0][0].get_shared_y_axes().join(self.a[0][0],self.a[0][1])
		self.a[1][0].get_shared_y_axes().join(self.a[1][0],self.a[1][1])
		self.a[0][0].get_shared_x_axes().join(self.a[0][0],self.a[1][0])
		self.a[0][1].get_shared_x_axes().join(self.a[0][1],self.a[1][1])

		## Set the ticks/labels so that they look nice
		for aa in self.a:
			for aaa in aa:
				for asp in ['top','bottom','left','right']:
					aaa.spines[asp].set_linewidth(1.0/self.canvas.devicePixelRatio())
				aaa.tick_params(labelsize=12./self.canvas.devicePixelRatio(),axis='both',direction='in',width=1.0/self.canvas.devicePixelRatio(),length=2./self.canvas.devicePixelRatio())
				aaa.tick_params(axis='both',which='major',length=4./self.canvas.devicePixelRatio())

		plt.setp(self.a[0][0].get_xticklabels(), visible=False)
		for aa in [self.a[0][1],self.a[1][1]]:
			aa.yaxis.tick_right()
			plt.setp(aa.get_yticklabels(),visible=False)
			plt.setp(aa.get_xticklabels(),visible=False)
			# aa.tick_params(axis='x', which='both',length=0)
		self.a[0][1].tick_params(axis='x', which='both',length=0)
		self.a[0][0].tick_params(axis='x', which='both',length=0)
		self.a[0][1].tick_params(axis='y',which='both',direction='in')
		self.a[1][1].tick_params(axis='y',which='both',direction='in')

		## Redraw everything
		self.update_axes()
		self.label_axes()
		self.f.tight_layout()
		offset1 = .08
		offset2 = 0.02
		offset3 = 0.14
		self.f.subplots_adjust(left=offset3,right=1.-offset2,top=1.-offset1,bottom=offset3,hspace=.03,wspace=0.015)

		self.canvas.draw()
		# self.draw()

		for i in range(self.gui.ncolors):
			## plot pre-truncated, kept, and post-truncated trajectory (Intensities)
			color = self.gui.prefs['channel_colors'][i]
			self.a[0][0].plot(np.random.rand(self.gui.data.d.shape[0]),color=color,ls=':',alpha=pb,lw=lw)
			self.a[0][0].plot(np.random.rand(self.gui.data.d.shape[0]),color=color,alpha=.8,lw=lw)
			self.a[0][0].plot(np.random.rand(self.gui.data.d.shape[0]),color=color,ls=':',alpha=pb,lw=lw)

			## Plot histograms of intensities
			self.a[0][1].plot(np.random.rand(100),color=color,alpha=.8,lw=1./ self.canvas.devicePixelRatio())

		## plot pre-truncated, kept, and post-truncated trajectory (E_{FRET})
		for i in range(1,self.gui.ncolors):
			if self.gui.ncolors == 2:
				color = 'blue'
			else:
				color = self.gui.prefs['channel_colors'][i]
			self.a[1][0].plot(np.random.rand(self.gui.data.d.shape[0]),color=color,ls=':',alpha=pb,lw=lw)
			self.a[1][0].plot(np.random.rand(self.gui.data.d.shape[0]),color=color,alpha=.8,lw=lw)
			self.a[1][0].plot(np.random.rand(self.gui.data.d.shape[0]),color=color,ls=':',alpha=pb,lw=lw)
			## Plot histograms of rel. intensities
			self.a[1][1].plot(np.random.rand(100),color=color,alpha=.8,lw=1./ self.canvas.devicePixelRatio())

		self.update_colors()

		self.update_blits()
		self.update_plots()

	def update_blits(self):
		[[[l.set_visible(False) for l in aaa.lines] for aaa in aa] for aa in self.a]
		self.f.canvas.draw()
		self.blit_bgs = [[self.f.canvas.copy_from_bbox(aaa.bbox) for aaa in aa] for aa in self.a]
		[[[l.set_visible(True) for l in aaa.lines] for aaa in aa] for aa in self.a]

	def initialize_hmm_plot(self):
		if len(self.a[1,0].lines) < 4:
			self.a[1,0].plot(np.random.rand(100),np.random.rand(100),color='k',lw=1.,alpha=.8)

	## Add axis labels to plots
	def label_axes(self):
		fs = 12./self.canvas.devicePixelRatio()

		self.a[0][0].set_ylabel(r'Intensity (a.u.)',fontsize=fs,va='top')
		if self.gui.ncolors == 2:
			self.a[1][0].set_ylabel(r'E$_{\rm{FRET}}$',fontsize=fs,va='top')
		else:
			self.a[1][0].set_ylabel(r'Relative Intensity',fontsize=fs,va='top')
		self.a[1][0].set_xlabel(r'Time (s)',fontsize=fs)
		self.a[1][1].set_xlabel(r'Probability',fontsize=fs)

		self.a[0][0].yaxis.set_label_coords(-.18, 0.5)
		self.a[1][0].yaxis.set_label_coords(-.18, 0.5)
		self.a[1][0].xaxis.set_label_coords(0.5, -.21)
		self.a[1][1].xaxis.set_label_coords(0.5, -.21)
