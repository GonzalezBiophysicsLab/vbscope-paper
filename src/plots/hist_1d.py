import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener
from PyQt5.QtWidgets import QPushButton
import multiprocessing as mp

default_prefs = {
	'fig_height':2.5,
	'fig_width':2.5,
	'label_fontsize':10.,
	'label_ticksize':8.,

	'fret_min':-.2,
	'fret_max':1.2,
	'fret_nbins':141,
	'fret_clip_low':-1.,
	'fret_clip_high':2.,
	'label_padding_left':.15,
	'label_padding_bottom':.15,
	'label_padding_top':.05,
	'label_padding_right':.05,
	'label_space':2.,

	'hist_type':'stepfilled',
	'hist_color':'steelblue',
	'hist_edgecolor':'black',
	'hist_log_y':False,
	'hist_force_ymax':True,
	'hist_ymax':2.5,
	'hist_ymin':0.0,

	'label_x_nticks':8,
	'label_y_nticks':6,

	'kde_bandwidth':0.0,

	'gmm_on':True,
	'hmm_on':True,
	'hmm_viterbi':False,
	'hmm_states':False,
	'hist_on':True,


	'vb_maxstates':8,
	'vb_prior_beta':0.25,
	'vb_prior_a':2.5,
	'vb_prior_b':0.01,
	'vb_prior_alpha':1.,
	'gmm_nrestarts':4,
	'ncpu':mp.cpu_count(),
	'gmm_threshold':1e-10,
	'gmm_maxiters':1000,

	'wiener_filter':False,
	'textbox_x':0.965,
	'textbox_y':0.9,
	'textbox_fontsize':10,
	'textbox_nmol':True
}



def setup(gui):

	fitvbbutton = QPushButton("VB gmm Fit")
	gui.popout_plots['plot_hist1d'].ui.buttonbox.insertWidget(2,fitvbbutton)
	fitvbbutton.clicked.connect(lambda x: fit_vb(gui))

	fitmlbutton = QPushButton("ML gmm Fit")
	gui.popout_plots['plot_hist1d'].ui.buttonbox.insertWidget(3,fitmlbutton)
	fitmlbutton.clicked.connect(lambda x: fit_ml(gui))

	recalcbutton = QPushButton("Recalculate")
	gui.popout_plots['plot_hist1d'].ui.buttonbox.insertWidget(1,recalcbutton)
	recalcbutton.clicked.connect(lambda x: recalc(gui))

	gui.popout_plots['plot_hist1d'].ui.gmm_result = None
	recalc(gui)




def fit_vb(gui):
	if not gui.data.d is None:
		gui.set_status('Compiling...')
		gui.app.processEvents()
		from ..supporting.hmms.vb_em_gmm import vb_em_gmm,vb_em_gmm_parallel
		gui.set_status('')

		prefs = gui.popout_plots['plot_hist1d'].ui.prefs

		fpb = gui.popout_plots['plot_hist1d'].ui.fpb.flatten()
		fpb = fpb[np.isfinite(fpb)]
		bad = np.bitwise_or((fpb < prefs['fret_clip_low']),(fpb > prefs['fret_clip_high']))
		fpb[bad] = np.random.uniform(low=prefs['fret_clip_low'],high=prefs['fret_clip_high'],size=int(bad.sum())) ## clip

		ll = np.zeros(prefs['vb_maxstates'])
		rs = [None for _ in range(ll.size)]

		from ..ui.ui_progressbar import progressbar
		prog = progressbar()
		prog.setRange(0,ll.size)
		prog.setWindowTitle('VB gmm Progress')
		prog.setLabelText('Number of States')
		# self.flag_running = True
		# prog.canceled.connect(self._cancel_run)
		prog.show()

		priors = np.array([gui.prefs[sss] for sss in ['vb_prior_beta','vb_prior_a','vb_prior_b','vb_prior_alpha']])

		for i in range(ll.size):
			r = vb_em_gmm_parallel(fpb,i+1,maxiters=prefs['gmm_maxiters'],threshold=prefs['gmm_threshold'],nrestarts=prefs['gmm_nrestarts'],prior_strengths=priors,ncpu=prefs['ncpu'])
			rs[i] = r
			ll[i] = r.likelihood[-1,0]
			prog.setValue(i+1)
			gui.app.processEvents()

		if np.all(np.isnan(ll)):
			gui.popout_plots['plot_hist1d'].ui.gmm_result = None
			gui.log("VB gmm failed: all lowerbounds are NaNs")
		else:
			n = np.nanargmax(ll)
			r = rs[n]
			r.type = 'vb'
			gui.popout_plots['plot_hist1d'].ui.gmm_result = r

			recalc(gui)
			# plot(gui)

			gui.log(r.report(),True)

def fit_ml(gui):
	if not gui.data.d is None:
		gui.set_status('Compiling...')
		gui.app.processEvents()
		from ..supporting.hmms.ml_em_gmm import ml_em_gmm
		gui.set_status('')

		prefs = gui.popout_plots['plot_hist1d'].ui.prefs

		success,nstates = gui.data.get_nstates()
		if success:
			fpb = gui.popout_plots['plot_hist1d'].ui.fpb.flatten()
			fpb = fpb[np.isfinite(fpb)]
			bad = np.bitwise_or((fpb < prefs['fret_clip_low']),(fpb > prefs['fret_clip_high']))
			fpb[bad] = np.random.uniform(low=prefs['fret_clip_low'],high=prefs['fret_clip_high'],size=int(bad.sum())) ## clip

			r = ml_em_gmm(fpb,nstates+1,maxiters=1000,threshold=1e-6)
			r.type = 'ml'
			gui.popout_plots['plot_hist1d'].ui.gmm_result = r

			recalc(gui)
			# plot(gui)

			gui.log(r.report(),True)

def draw_gmm(gui):
	try:
		popplot = gui.popout_plots['plot_hist1d'].ui

		r = popplot.gmm_result
		x = popplot.gmm_x
		tot = popplot.gmm_tot
		ys = popplot.gmm_ys

		if r.type == 'vb':
			for i in range(len(ys)):
				y = ys[i]
				popplot.ax[0].plot(x,y,color='k',lw=1,alpha=.8,ls='--')
		elif r.type == 'ml':
			for i in range(len(ys) - 1): ## ignore the outlier class
				y = ys[i]
				popplot.ax[0].plot(x,y,color='k',lw=1,alpha=.8,ls='--')
			y = ys[-1]
			popplot.ax[0].plot(x,y,color='k',lw=1,alpha=.8,ls='--')

		popplot.ax[0].plot(x,tot,color='k',lw=2,alpha=.8)
		popplot.f.canvas.draw()
	except:
		pass

def draw_hmm(gui):
	try:
		popplot = gui.popout_plots['plot_hist1d'].ui

		r = gui.data.hmm_result
		x = popplot.hmm_x
		tot = popplot.hmm_tot
		ys = popplot.hmm_ys

		if r.type == 'consensus vbfret':
			for y in ys:
				popplot.ax[0].plot(x,y,color='k',lw=1,alpha=.8,ls='--')

		popplot.ax[0].plot(x,tot,color='k',lw=2,alpha=.8)
		popplot.f.canvas.draw()
	except:
		pass

def normal(x,m,v):
	return 1./np.sqrt(2.*np.pi*v)*np.exp(-.5/v*(x-m)**2.)

def studentt(x,a,b,k,m):
	from scipy.special import gammaln
	lam = a*k/(b*(k+1.))
	lny = -.5*np.log(np.pi)
	lny += gammaln((2.*a+1.)/2.) - gammaln((a))
	lny += .5*np.log(lam/(2.*a))
	lny += -.5*(2.*a+1)*np.log(1.+lam*(x-m)**2./(2.*a))
	return np.exp(lny)


def kde(x,d,bw=None):
	from scipy.stats import gaussian_kde
	if bw is None:
		kernel = gaussian_kde(d)
	else:
		kernel = gaussian_kde(d,bw)
	y = kernel(x)
	return y


def recalc(gui):
	popplot = gui.popout_plots['plot_hist1d'].ui

	## Data
	if popplot.prefs['hmm_viterbi']:
		gui.popout_plots['plot_hist1d'].ui.fpb = gui.data.get_viterbi_data(signal=True)
	else:
		fpb = gui.data.get_plot_data()[0].copy()
		if popplot.prefs['wiener_filter'] is True:
			for i in range(fpb.shape[0]):
				cut = np.isfinite(fpb[i])
			try:
				fpb[i][cut] = wiener(fpb[i][cut])
			except:
				pass
		gui.popout_plots['plot_hist1d'].ui.fpb = fpb

	## GMM
	if not popplot.gmm_result is None:
		r = popplot.gmm_result
		x = np.linspace(popplot.prefs['fret_min'],popplot.prefs['fret_max'],1001)

		tot = np.zeros_like(x)
		ys = []
		if r.type == 'vb':
			for i in range(r.mu.size):
				y = r.ppi[i]*studentt(x,r.a[i],r.b[i],r.beta[i],r.m[i])
				tot += y
				ys.append(y)
		elif r.type == 'ml':
			for i in range(r.mu.size - 1): ## ignore the outlier class
				y = r.ppi[i]*normal(x,r.mu[i],r.var[i])
				tot += y
				ys.append(y)
			y = r.ppi[-1]*(x*0. + r.mu[-1])
			tot += y
			ys.append(y)

		gui.popout_plots['plot_hist1d'].ui.gmm_x = x
		gui.popout_plots['plot_hist1d'].ui.gmm_tot = tot
		gui.popout_plots['plot_hist1d'].ui.gmm_ys = ys

	## HMM -- this needs to be standardized... kind of a mess
	if not gui.data.hmm_result is None:
		r = gui.data.hmm_result

		x = np.linspace(popplot.prefs['fret_min'],popplot.prefs['fret_max'],1001)
		tot = np.zeros_like(x)
		ys = []
		if r.type == 'consensus vbfret':
			rr = r.result
			for i in range(rr.m.size):
				if popplot.prefs['hmm_states']:
					y = rr.ppi[i]*normal(x,rr.m[i],1./rr.beta[i])
				else:
					y = rr.ppi[i]*studentt(x,rr.a[i],rr.b[i],rr.beta[i],rr.m[i])
				tot += y
				ys.append(y)
		elif r.type == 'vb' or r.type == 'ml':
			nn = 0.
			if not popplot.prefs['hmm_states']:
				for j in range(len(r.results)):
					rr = r.results[j]
					nn += rr.r.shape[0]
					for i in range(rr.mu.size):
						if r.type == 'ml':
							tot += rr.ppi[i]*normal(x,rr.mu[i],rr.var[i]) * rr.r.shape[0]
						else:
							tot += rr.ppi[i]*studentt(x,rr.a[i],rr.b[i],rr.beta[i],rr.m[i]) * rr.r.shape[0]
				tot /= nn
			else:
				if popplot.prefs['hmm_viterbi']:
					v = gui.data.get_viterbi_data(signal=True).flatten()
					v = v[np.isfinite(v)]
					if popplot.prefs['kde_bandwidth'] != 0.0:
						try:
							tot = kde(x,v,popplot.prefs['kde_bandwidth'])
						except:
							tot = kde(x,v)
					else:
						tot = kde(x,v)

				else:
					for j in range(len(r.results)):
						rr = r.results[j]
						nn += rr.r.shape[0]
						for i in range(rr.mu.size):
							if r.type == 'ml':
								tot += rr.ppi[i]*normal(x,rr.mu[i],rr.var[i]/np.sqrt(rr.ppi[i]*rr.r.shape[0])) * rr.r.shape[0]
							else:
								tot += rr.ppi[i]*normal(x,rr.m[i],1./rr.beta[i]) * rr.r.shape[0]
					tot /= nn

		gui.popout_plots['plot_hist1d'].ui.hmm_x = x
		gui.popout_plots['plot_hist1d'].ui.hmm_tot = tot
		gui.popout_plots['plot_hist1d'].ui.hmm_ys = ys
	plot(gui)


def plot(gui):
	popplot = gui.popout_plots['plot_hist1d'].ui
	popplot.ax[0].cla()
	popplot.resize_fig()
	gui.app.processEvents()

	if gui.ncolors == 2:
		fpb = gui.popout_plots['plot_hist1d'].ui.fpb

		if popplot.prefs['hist_on']:
			try:
				popplot.ax[0].hist(popplot.fpb.flatten(),bins=popplot.prefs['fret_nbins'],range=(popplot.prefs['fret_min'],popplot.prefs['fret_max']),histtype=popplot.prefs['hist_type'],alpha=.8,density=True,color=popplot.prefs['hist_color'],edgecolor=popplot.prefs['hist_edgecolor'],log=popplot.prefs['hist_log_y'])
			except:
				popplot.ax[0].hist(fpb.flatten(),bins=popplot.prefs['fret_nbins'],range=(popplot.prefs['fret_min'],popplot.prefs['fret_max']),histtype='stepfilled',alpha=.8,density=True,color='steelblue',edgecolor='k',log=popplot.prefs['hist_log_y'])
		else:
			if popplot.prefs['hist_log_y']:
				popplot.ax[0].set_yscale('log')

		ylim = popplot.ax[0].get_ylim()
		if popplot.prefs['hmm_on'] and not gui.data.hmm_result is None:
			draw_hmm(gui)
		if popplot.prefs['gmm_on'] and not popplot.gmm_result is None:
			draw_gmm(gui)

		popplot.ax[0].set_xlim(popplot.prefs['fret_min'],popplot.prefs['fret_max'])
		if not popplot.prefs['hist_on']:
			popplot.ax[0].set_ylim(*ylim)

		if popplot.prefs['hist_force_ymax']:
			popplot.ax[0].set_ylim(popplot.prefs['hist_ymin'],popplot.prefs['hist_ymax'])
			popplot.ax[0].set_yticks(np.linspace(popplot.prefs['hist_ymin'],popplot.prefs['hist_ymax'],popplot.prefs['label_y_nticks']))

		popplot.ax[0].set_xticks(np.linspace(popplot.prefs['fret_min'],popplot.prefs['fret_max'],popplot.prefs['label_x_nticks']))

		popplot.ax[0].set_xlabel(r'$\rm E_{\rm FRET}(t)$',fontsize=popplot.prefs['label_fontsize']/gui.plot.canvas.devicePixelRatio(),labelpad=popplot.prefs['label_space'])
		popplot.ax[0].set_ylabel('Probability',fontsize=popplot.prefs['label_fontsize']/gui.plot.canvas.devicePixelRatio(),labelpad=popplot.prefs['label_space'])

		for asp in ['top','bottom','left','right']:
			popplot.ax[0].spines[asp].set_linewidth(1.0/gui.plot.canvas.devicePixelRatio())
		popplot.f.subplots_adjust(left=popplot.prefs['label_padding_left'],bottom=popplot.prefs['label_padding_bottom'],top=1.-popplot.prefs['label_padding_top'],right=1.-popplot.prefs['label_padding_right'])

		bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=1./gui.plot.canvas.devicePixelRatio())
		lstr = 'N = %d'%(popplot.fpb.shape[0])

		popplot.ax[0].annotate(lstr,xy=(popplot.prefs['textbox_x'],popplot.prefs['textbox_y']),xycoords='axes fraction',ha='right',color='k',bbox=bbox_props,fontsize=popplot.prefs['textbox_fontsize']/gui.plot.canvas.devicePixelRatio())

		popplot.f.canvas.draw()
