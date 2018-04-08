import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener
from PyQt5.QtWidgets import QPushButton
import multiprocessing as mp

default_prefs = {
	'fret_min':-.25,
	'fret_max':1.25,
	'fret_nbins':161,

	'hist_type':'stepfilled',
	'hist_color':'steelblue',
	'hist_edgecolor':'black',
	'hist_log_y':False,

	'label_x_nticks':7,

	'overlay_GMM':True,
	'overlay_HMM':True,

	'vb_maxstates':8,
	'vb_prior_beta':0.25,
	'vb_prior_a':2.5,
	'vb_prior_b':0.01,
	'vb_prior_alpha':1.,
	'vb_prior_pi':1.,
	'gmm_nrestarts':4,
	'ncpu':mp.cpu_count(),
	'gmm_threshold':1e-10,
	'gmm_maxiters':1000,

	'wiener_filter':False,
	'textbox_x':0.95,
	'textbox_y':0.93,
	'textbox_fontsize':8,
	'textbox_nmol':True
}

def setup(gui):

	fitvbbutton = QPushButton("VB GMM Fit")
	gui.popout_plots['plot_hist1d'].ui.buttonbox.insertWidget(2,fitvbbutton)
	fitvbbutton.clicked.connect(lambda x: fit_vb(gui))

	fitmlbutton = QPushButton("ML GMM Fit")
	gui.popout_plots['plot_hist1d'].ui.buttonbox.insertWidget(3,fitmlbutton)
	fitmlbutton.clicked.connect(lambda x: fit_ml(gui))

	gui.popout_plots['plot_hist1d'].ui.gmm_result = None


def fit_vb(gui):
	if not gui.data.d is None:
		gui.set_status('Compiling...')
		gui.app.processEvents()
		from ..supporting.hmms.vb_em_gmm import vb_em_gmm,vb_em_gmm_parallel
		gui.set_status('')

		prefs = gui.popout_plots['plot_hist1d'].ui.prefs

		fpb = get_data(gui).flatten()
		fpb = fpb[np.isfinite(fpb)]
		bad = np.bitwise_or((fpb < -1.),(fpb > 2))
		fpb[bad] = np.random.uniform(low=-1,high=2,size=int(bad.sum())) ## clip

		ll = np.zeros(prefs['vb_maxstates'])
		rs = [None for _ in range(ll.size)]

		from ..ui.ui_progressbar import progressbar
		prog = progressbar()
		prog.setRange(0,ll.size)
		prog.setWindowTitle('VB GMM Progress')
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
			gui.log("VB GMM failed: all lowerbounds are NaNs")
		else:
			n = np.nanargmax(ll)
			r = rs[n]
			r.type = 'vb'
			gui.popout_plots['plot_hist1d'].ui.gmm_result = r

			plot(gui)

			gui.log(r.report(),True)

def fit_ml(gui):
	if not gui.data.d is None:
		gui.set_status('Compiling...')
		gui.app.processEvents()
		from ..supporting.hmms.ml_em_gmm import ml_em_gmm
		gui.set_status('')

		success,nstates = gui.data.get_nstates()
		if success:
			fpb = get_data(gui).flatten()
			fpb = fpb[np.isfinite(fpb)]
			bad = np.bitwise_or((fpb < -1.),(fpb > 2))
			fpb[bad] = np.random.uniform(low=-1,high=2,size=int(bad.sum())) ## clip

			r = ml_em_gmm(fpb,nstates+1,maxiters=1000,threshold=1e-6)
			r.type = 'ml'
			gui.popout_plots['plot_hist1d'].ui.gmm_result = r

			plot(gui)

			gui.log(r.report(),True)

def draw_gmm(gui):
	popplot = gui.popout_plots['plot_hist1d'].ui
	r = popplot.gmm_result

	x = np.linspace(popplot.prefs['fret_min'],popplot.prefs['fret_max'],1001)

	tot = np.zeros_like(x)
	if r.type == 'vb':
		for i in range(r.mu.size):
			y = r.ppi[i]*studentt(x,r.a[i],r.b[i],r.beta[i],r.m[i])
			tot += y
			popplot.ax[0].plot(x,y,color='k',lw=1,alpha=.8,ls='--')
	elif r.type == 'ml':
		for i in range(r.mu.size - 1): ## ignore the outlier class
			y = r.ppi[i]*normal(x,r.mu[i],r.var[i])
			tot += y
			popplot.ax[0].plot(x,y,color='k',lw=1,alpha=.8,ls='--')
		y = r.ppi[-1]*(x*0. + r.mu[-1])
		tot += y
		popplot.ax[0].plot(x,y,color='k',lw=1,alpha=.8,ls='--')


	popplot.ax[0].plot(x,tot,color='k',lw=2,alpha=.8)
	popplot.f.canvas.draw()

def draw_hmm(gui):
	popplot = gui.popout_plots['plot_hist1d'].ui

	r = gui.data.hmm_result

	x = np.linspace(popplot.prefs['fret_min'],popplot.prefs['fret_max'],1001)
	tot = np.zeros_like(x)
	if r.type == 'consensus vbfret':
		rr = r.result
		for i in range(rr.m.size):
			y = rr.ppi[i]*studentt(x,rr.a[i],rr.b[i],rr.beta[i],rr.m[i])
			tot += y
			popplot.ax[0].plot(x,y,color='k',lw=1,alpha=.8,ls='--')
	elif r.type == 'vb':
		nn = 0.
		for j in range(len(r.results)):
			rr = r.results[j]
			nn += rr.r.shape[0]
			for i in range(rr.m.size):
				tot += rr.ppi[i]*studentt(x,rr.a[i],rr.b[i],rr.beta[i],rr.m[i]) * rr.r.shape[0]
		tot /= nn
	elif r.type == 'ml':
		nn = 0.
		for j in range(len(r.results)):
			rr = r.results[j]
			nn += rr.r.shape[0]
			for i in range(rr.mu.size):
				tot += rr.ppi[i]*normal(x,rr.mu[i],rr.var[i]) * rr.r.shape[0]
		tot /= nn
	popplot.ax[0].plot(x,tot,color='k',lw=2,alpha=.8)
	popplot.f.canvas.draw()

def get_data(gui):
	fpb = gui.data.get_plot_data()[0]
	if gui.popout_plots['plot_hist1d'].ui.prefs['wiener_filter'] is True:
		for i in range(fpb.shape[0]):
			cut = np.isfinite(fpb[i])
			try:
				fpb[i][cut] = wiener(fpb[i][cut])
			except:
				pass
	return fpb

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

def plot(gui):
	popplot = gui.popout_plots['plot_hist1d'].ui
	popplot.ax[0].cla()
	popplot.resize_fig()
	gui.app.processEvents()

	if gui.ncolors == 2:
		fpb = get_data(gui)

		# plt.hist(f.flatten(),bins=181,range=(-.4,1.4),histtype='stepfilled',alpha=.8,normed=True)
		try:
			popplot.ax[0].hist(fpb.flatten(),bins=popplot.prefs['fret_nbins'],range=(popplot.prefs['fret_min'],popplot.prefs['fret_max']),histtype=popplot.prefs['hist_type'],alpha=.8,density=True,color=popplot.prefs['hist_color'],edgecolor=popplot.prefs['hist_edgecolor'],log=popplot.prefs['hist_log_y'])
		except:
			popplot.ax[0].hist(fpb.flatten(),bins=popplot.prefs['fret_nbins'],range=(popplot.prefs['fret_min'],popplot.prefs['fret_max']),histtype='stepfilled',alpha=.8,density=True,color='steelblue',log=popplot.prefs['hist_log_y'])

		ylim = popplot.ax[0].get_ylim()
		if popplot.prefs['overlay_HMM'] and not gui.data.hmm_result is None:
			draw_hmm(gui)
		if popplot.prefs['overlay_GMM'] and not popplot.gmm_result is None:
			draw_gmm(gui)

		popplot.ax[0].set_xlim(popplot.prefs['fret_min'],popplot.prefs['fret_max'])
		popplot.ax[0].set_ylim(*ylim)

		popplot.ax[0].set_xticks(np.linspace(popplot.prefs['fret_min'],popplot.prefs['fret_max'],popplot.prefs['label_x_nticks']))

		popplot.ax[0].set_xlabel(r'$\rm E_{\rm FRET}(t)$',fontsize=popplot.prefs['label_fontsize']/gui.plot.canvas.devicePixelRatio())
		popplot.ax[0].set_ylabel('Probability',fontsize=popplot.prefs['label_fontsize']/gui.plot.canvas.devicePixelRatio())

		for asp in ['top','bottom','left','right']:
			popplot.ax[0].spines[asp].set_linewidth(1.0/gui.plot.canvas.devicePixelRatio())
		popplot.f.subplots_adjust(left=.05+popplot.prefs['label_padding'],bottom=.05+popplot.prefs['label_padding'],top=.95,right=.95)

		bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=1./gui.plot.canvas.devicePixelRatio())
		lstr = 'N = %d'%(fpb.shape[0])

		popplot.ax[0].annotate(lstr,xy=(popplot.prefs['textbox_x'],popplot.prefs['textbox_y']),xycoords='axes fraction',ha='right',color='k',bbox=bbox_props,fontsize=popplot.prefs['textbox_fontsize']/gui.plot.canvas.devicePixelRatio())

		popplot.f.canvas.draw()
