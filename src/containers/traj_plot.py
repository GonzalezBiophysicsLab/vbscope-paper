from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from matplotlib.widgets import  RectangleSelector
from PyQt5.QtWidgets import QSizePolicy,QVBoxLayout,QWidget

import numpy as np

default_prefs = {
	'plot_fret_min':float(-.25),
	'plot_fret_max':float(1.25),
	'plot_intensity_min':float(-1000.0),
	'plot_intensity_max':float(10000.0),
	'plot_filter':False,
	'plot_channel_colors':["#0EA52F","#EE0000","cyan","purple"],
	'plot_downsample':1,
	'plot_fret_color':'#023bf9',
	'plot_line_linewidth':0.7,
	'plot_hist_linewidth':1.0,
	'plot_hist_show':True,
	'plot_line_alpha_pb':0.25,
	'plot_axes_linewidth':1.0,
	'plot_axes_topright':False,
	'plot_tick_fontsize':8.0,
	'plot_tick_length_minor':2.0,
	'plot_tick_length_major':4.0,
	'plot_tick_linewidth':1.0,
	'plot_tick_direction':'out',
	'plot_subplots_left':0.125,
	'plot_subplots_right':0.99,
	'plot_subplots_top':0.99,
	'plot_subplots_bottom':0.155,
	'plot_subplots_hspace':0.04,
	'plot_subplots_wspace':0.03,
	'plot_line_alpha':0.9,
	'plot_viterbi_color':'k',
	'plot_viterbi_linewidth':0.7,
	'plot_viterbi_alpha':0.9,
	'plot_label_fontsize':8.0,
	'plot_ylabel_offset':-0.165,
	'plot_xlabel_offset':-0.25,
	'plot_font':'Arial',
	'plot_xlabel_text1':'Time (s)',
	'plot_xlabel_text2':'Probability',
	'plot_ylabel_text1':r'Intensity (a.u.)',
	'plot_ylabel_text2':r'E$_{\rm{FRET}}$',
	'plot_fret_pbzero':True,
	'plot_fret_nticks':7,
	'plot_intensity_nticks':6,
	'plot_time_rotate':0.0,
	'plot_time_min':0.0,
	'plot_time_max':1.0,
	'plot_time_nticks':6,
	'plot_time_offset':0.0,
	'plot_time_decimals':0,
	'plot_intensity_decimals':0,
	'plot_fret_decimals':2,
	'normalize_intensities':False
}


class traj_plot_container():
	'''
	.f - figure
	.ax - axis
	.toolbar - MPL toolbar
	'''
	def __init__(self,gui):
		self.gui = gui
		if self.gui.prefs.count('plot_fret_min') == 0:
			self.gui.prefs.add_dictionary(default_prefs)

		self.f,self.a = plt.subplots(2,2,gridspec_kw={'width_ratios':[6,1]},figsize=(6.5,5))
		self.canvas = FigureCanvas(self.f)
		self.f.set_dpi(self.f.get_dpi()/self.canvas.devicePixelRatio())

		# sizePolicy = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
		sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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
		try:
			self.canvas.draw()
		except:
			pass

	## Read in y-min and y-max values, then update the plot
	def update_minmax(self):
		# self.initialize_plots()
		try:
			self.a[0][0].set_ylim(self.gui.prefs['plot_intensity_min'],self.gui.prefs['plot_intensity_max'])
		except:
			self.a[0][0].set_ylim(-1000.,10000.)
		self.canvas.draw()

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
			if not self.gui.prefs['plot_fret_pbzero']:
				self.a[1][0].lines[3*i+2].set_data(t[pbtime:],rel[i,pbtime:])
			else:
				self.a[1][0].lines[3*i+2].set_data(t[pbtime:],np.zeros_like(rel[i,pbtime:]))

	def calc_trajectory(self):
		intensities = self.gui.data.d[self.index].copy()
		if self.gui.prefs['plot_filter'] is True:
			for i in range(intensities.shape[0]):
				try:
					intensities[i] = self.gui.data.filter(intensities[i])
				except:
					pass

		bts = self.gui.prefs['bleedthrough'].reshape((4,4))
		for i in range(self.gui.ncolors):
			for j in range(self.gui.ncolors):
				intensities[j] -= bts[i,j]*intensities[i]

		if self.gui.prefs['normalize_intensities']:
			q = intensities.sum(0)+50.0
			q /= q[0]
			intensities /= q[None,:]


		t = np.arange(intensities.shape[1])*self.gui.prefs['tau'] + self.gui.prefs['plot_time_offset']

		downsample = int(self.gui.prefs['plot_downsample'])
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
		yminmax = np.array((self.gui.prefs['plot_intensity_min'],self.gui.prefs['plot_intensity_max']))
		for i in range(self.gui.ncolors):
			if pretime < pbtime:
				hy,hx = np.histogram(intensities[i,pretime:pbtime],range=yminmax,bins=int(np.sqrt(pbtime-pretime)))
			else:
				hy = np.zeros(100)
				hx = np.linspace(yminmax[0],yminmax[1],101)
			hy = np.append(np.append(0.,hy),0.)
			hx = np.append(np.append(hx[0],.5*(hx[1:]+hx[:-1])),hx[-1])
			hymaxes.append(hy.max())
			intensity_hists.append([hx,hy])

		for i in range(self.gui.ncolors-1):
			if pretime < pbtime:
				hy,hx = np.histogram(rel[i,pretime:pbtime],range=(self.gui.prefs['plot_fret_min'],self.gui.prefs['plot_fret_max']),bins=int(np.sqrt(pbtime-pretime)))
			else:
				hy = np.zeros(100)
				hx = np.linspace(self.gui.prefs['plot_intensity_min'],self.gui.prefs['plot_intensity_max'],101)
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
			if self.gui.data.hmm_result.type == 'consensus vbfret':
				vitpath = self.gui.data.hmm_result.result.viterbi[ii]
				if self.gui.prefs['hmm_binding_expt'] is True:
					state_means = np.array((0.,1.))
				else:
					state_means = self.gui.data.hmm_result.result.m
			elif self.gui.data.hmm_result.type == 'vb' or self.gui.data.hmm_result.type == 'ml':
				vitpath = self.gui.data.hmm_result.results[ii].viterbi
				state_means = self.gui.data.hmm_result.results[ii].mu

		else:
			state_means = None
			vitpath = None
		return state_means,vitpath

	def update_axis_limits(self):
		self.a[1][0].set_xlim(self.gui.prefs['plot_time_min'],self.gui.prefs['plot_time_max'])
		self.a[0][0].set_ylim(self.gui.prefs['plot_intensity_min'],self.gui.prefs['plot_intensity_max'])
		self.a[0][1].set_xlim(0.01, 1.25)
		self.a[1][0].set_ylim(self.gui.prefs['plot_fret_min'],self.gui.prefs['plot_fret_max'])
		self.a[1][1].set_xlim(self.a[0][1].get_xlim())
		self.update_decimals()

	def update_decimals(self):
		fd = {'rotation':self.gui.prefs['plot_time_rotate'], 'ha':'center'}
		if fd['rotation'] != 0: fd['ha'] = 'right'
		xt = self.a[1][0].get_xticks()
		self.a[1][0].set_xticklabels(["{0:.{1}f}".format(x,self.gui.prefs['plot_time_decimals']) for x in xt],fontdict=fd)

		fd = {}
		yt = self.a[1][0].get_yticks()
		self.a[1][0].set_yticklabels(["{0:.{1}f}".format(y,self.gui.prefs['plot_fret_decimals']) for y in yt],fontdict=fd)
		yt = self.a[0][0].get_yticks()
		self.a[0][0].set_xticklabels(["{0:.{1}f}".format(y,self.gui.prefs['plot_intensity_decimals']) for y in yt],fontdict=fd)

	## Plot current trajectory
	def update_plots(self):
		try:
			self.canvas.blockSignals(True)
			if self.flag_arm:
				self.update_blits()
				self.flag_arm = False
			[[self.f.canvas.restore_region(bbb) for bbb in bb] for bb in self.blit_bgs]

			if type(self.gui.prefs['plot_hist_show']) is bool:
				for i in range(2):
					self.a[i][1].set_visible(self.gui.prefs['plot_hist_show'])

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


			self.update_axis_limits()
			self.update_axis_geometry()
			self.update_ticks()
			self.update_axis_labels()
			self.update_lines()

			[[[aaa.draw_artist(l) for l in aaa.lines] for aaa in aa] for aa in self.a]
			[[self.f.canvas.blit(aaa.bbox) for aaa in aa] for aa in self.a]


			self.canvas.update()
			self.canvas.flush_events()

			# yl = self.a[1][0].get_ylim()
			# if self.a[0][0].get_xlim()[1] != self.gui.data.d.shape[2]*self.gui.prefs['tau'] or yl[0] != self.gui.prefs['plot_fret_min'] or yl[1] != self.gui.prefs['plot_fret_max']:
			# if 1:



			self.draw()
			self.canvas.blockSignals(False)
		except:
			pass

	def update_ticks(self):
		dpr = self.canvas.devicePixelRatio()

		## Set the ticks/labels so that they look nice
		for aa in self.a:
			for aaa in aa:
				for asp in ['top','bottom','left','right']:
					aaa.spines[asp].set_linewidth(self.gui.prefs['plot_axes_linewidth']/dpr)
					if asp in ['top','right']:
						aaa.spines[asp].set_visible(self.gui.prefs['plot_axes_topright'])

				tickdirection = self.gui.prefs['plot_tick_direction']
				if not tickdirection in ['in','out']: tickdirection = 'in'
				aaa.tick_params(labelsize=self.gui.prefs['plot_tick_fontsize']/dpr, axis='both', direction=tickdirection , width=self.gui.prefs['plot_tick_linewidth']/dpr, length=self.gui.prefs['plot_tick_length_minor']/dpr)
				aaa.tick_params(axis='both',which='major',length=self.gui.prefs['plot_tick_length_major']/dpr)
				for label in aaa.get_xticklabels():
					label.set_family(self.gui.prefs['plot_font'])
				for label in aaa.get_yticklabels():
					label.set_family(self.gui.prefs['plot_font'])

		plt.setp(self.a[0][0].get_xticklabels(), visible=False)
		for aa in [self.a[0][1],self.a[1][1]]:
			plt.setp(aa.get_yticklabels(),visible=False)
			plt.setp(aa.get_xticklabels(),visible=False)
			aa.set_xticks(())
			aa.set_yticks(())
		self.a[0][0].tick_params(axis='x', which='both',length=0)

		self.a[0][0].set_yticks(self.figure_out_ticks(self.gui.prefs['plot_intensity_min'],self.gui.prefs['plot_intensity_max'],self.gui.prefs['plot_intensity_nticks']))
		self.a[1][0].set_yticks(self.figure_out_ticks(self.gui.prefs['plot_fret_min'],self.gui.prefs['plot_fret_max'],self.gui.prefs['plot_fret_nticks']))
		self.a[1][0].set_xticks(self.figure_out_ticks(self.gui.prefs['plot_time_min'],self.gui.prefs['plot_time_max'],self.gui.prefs['plot_time_nticks']))

	def figure_out_ticks(self,ymin,ymax,nticks):
		m = nticks
		if m <= 0: return ()
		if ymax <= ymin: return ()
		delta = ymax-ymin

		d = 10.0**(np.floor(np.log10(delta)))
		ind = np.arange(1,10)
		ind = np.concatenate((1./ind[::-1],ind))
		di = d*ind
		for i in range(ind.size):
			if np.floor(delta/di[i]) < m:
				s = di[i]
				break
		y0 = np.ceil(ymin/s)*s
		delta = ymax - y0

		d = 10.0**(np.floor(np.log10(delta)))
		ind = np.arange(1,10)
		ind = np.concatenate((1./ind[::-1],ind))
		di = d*ind
		for i in range(ind.size):
			if np.floor(delta/di[i]) < m:
				s = di[i]
				break
		y0 = np.ceil(ymin/s)*s
		delta = ymax - y0
		n = np.floor(delta/s+1e-10)
		return y0 + np.arange(n+1)*s


	def update_axis_geometry(self):
		self.f.tight_layout()

		self.f.subplots_adjust(left=self.gui.prefs['plot_subplots_left'],right=self.gui.prefs['plot_subplots_right'],top=self.gui.prefs['plot_subplots_top'],bottom=self.gui.prefs['plot_subplots_bottom'],hspace=self.gui.prefs['plot_subplots_hspace'],wspace=self.gui.prefs['plot_subplots_wspace'])

	def update_line(self,l,color,alpha,linewidth):
		l.set_color(color)
		l.set_alpha(alpha)
		l.set_linewidth(linewidth)

	def update_lines(self):
		p = self.gui.prefs
		dpr = self.canvas.devicePixelRatio()
		lw = p['plot_line_linewidth']/dpr
		hw = p['plot_hist_linewidth']/dpr

		la = p['plot_line_alpha']
		lapb = p['plot_line_alpha_pb']
		alphas = [lapb,la,lapb]

		## Intensities
		for i in range(self.gui.ncolors):
			color = p['plot_channel_colors'][i]
			for j,alpha in zip(range(3),alphas):
				self.update_line(self.a[0][0].lines[3*i+j], color, alpha, lw)
			self.update_line(self.a[0][1].lines[i], color, p['plot_line_alpha'], hw)

		## Rel. Intensities
		for i in range(self.gui.ncolors-1):
			if self.gui.ncolors == 2:
				color = self.gui.prefs['plot_fret_color']
			else:
				color = self.gui.prefs['plot_channel_colors'][i+1]
			for j,alpha in zip(range(3),alphas):
				self.update_line(self.a[1][0].lines[3*i+j], color, alpha, lw)
			self.update_line(self.a[1][1].lines[i], color, p['plot_line_alpha'], hw)

		if len(self.a[1,0].lines)%4==0:
			for i in range(self.gui.ncolors-1):
				color = p['plot_viterbi_color']
				alpha = p['plot_viterbi_alpha']
				lw = p['plot_viterbi_linewidth']/dpr
				self.update_line(self.a[1,0].lines[-(1+i)], color, alpha, lw)

	def update_blits(self):
		[[[l.set_visible(False) for l in aaa.lines] for aaa in aa] for aa in self.a]
		self.f.canvas.draw()
		self.blit_bgs = [[self.f.canvas.copy_from_bbox(aaa.bbox) for aaa in aa] for aa in self.a]
		[[[l.set_visible(True) for l in aaa.lines] for aaa in aa] for aa in self.a]

	## Add axis labels to plots
	def update_axis_labels(self):
		fs = self.gui.prefs['plot_label_fontsize']/self.canvas.devicePixelRatio()
		font = {
			'family': self.gui.prefs['plot_font'],
			'size': fs,
			'va':'top'
		}

		self.a[0][0].set_ylabel(self.gui.prefs['plot_ylabel_text1'],fontdict=font)
		self.a[1][0].set_ylabel(self.gui.prefs['plot_ylabel_text2'],fontdict=font)
		self.a[1][0].set_xlabel(self.gui.prefs['plot_xlabel_text1'],fontdict=font)
		self.a[1][1].set_xlabel(self.gui.prefs['plot_xlabel_text2'],fontdict=font)

		self.a[0][0].yaxis.set_label_coords(self.gui.prefs['plot_ylabel_offset'], 0.5)
		self.a[1][0].yaxis.set_label_coords(self.gui.prefs['plot_ylabel_offset'], 0.5)
		self.a[1][0].xaxis.set_label_coords(0.5, self.gui.prefs['plot_xlabel_offset'])
		self.a[1][1].xaxis.set_label_coords(0.5, self.gui.prefs['plot_xlabel_offset'])

	## Plot initial data to set aesthetics
	def initialize_plots(self):
		## clear everything
		[[aaa.cla() for aaa in aa] for aa in self.a]

		## Make it so that certain plots zoom together
		self.a[0][0].get_shared_y_axes().join(self.a[0][0],self.a[0][1])
		self.a[1][0].get_shared_y_axes().join(self.a[1][0],self.a[1][1])
		self.a[0][0].get_shared_x_axes().join(self.a[0][0],self.a[1][0])
		self.a[0][1].get_shared_x_axes().join(self.a[0][1],self.a[1][1])

		## Redraw everything
		self.gui.prefs['plot_time_max'] = self.gui.data.d.shape[2]*self.gui.prefs['tau']
		self.update_ticks()
		self.update_axis_limits()
		self.update_axis_labels()
		self.update_axis_geometry()

		self.canvas.draw()

		for i in range(self.gui.ncolors):
			for ls in [':','-',':']:
				self.a[0][0].plot(np.random.rand(self.gui.data.d.shape[0]), ls=ls)
				if i > 0:
					self.a[1][0].plot(np.random.rand(self.gui.data.d.shape[0]), ls=ls)

		for i in range(self.gui.ncolors):
			self.a[0][1].plot(np.random.rand(100))
			if i > 0:
				self.a[1][1].plot(np.random.rand(100))

		self.update_lines()

		self.update_blits()
		self.update_plots()

	def initialize_hmm_plot(self):
		self.gui.update_display_traces()
		if len(self.a[1,0].lines) < 4:
			self.a[1,0].plot(np.random.rand(100),np.random.rand(100))
