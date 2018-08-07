import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit,minimize
from PyQt5.QtWidgets import QPushButton, QComboBox
from matplotlib import ticker
import numba as nb

default_prefs = {
'subplots_left':.15,
'subplots_top':.95,
'fig_width':4.0,
'fig_height':2.5,
'xlabel_offset':-.15,
'ylabel_offset':-.15,

'time_scale':'log',
'time_dt':1.0,
'time_nticks':5,
'time_min':0.0,
'time_max':2000.0,

'acorr_nticks':6,
'acorr_min':-0.1,
'acorr_max':1.0,
# 'acorr_filter':1.0,
# 'acorr_highpass':0.,

'power_nticks':6,
'power_min':.1,
'power_max':100.0,

'line_color':'blue',
'line_linewidth':1,
'line_alpha':0.9,

'fill_alpha':0.3,
'fill_color':'blue',

'xlabel_rotate':0.,
'ylabel_rotate':0.,
'xlabel_text1':r'Time(s)',
'ylabel_text1':r'Autocorrelation Function',
'xlabel_text2':r'Frequency (s$^{-1}$)',
'ylabel_text2':r'Power Spectrum',
'xlabel_decimals':2,
'ylabel_decimals':2,

'textbox_x':0.95,
'textbox_y':0.93,
'textbox_fontsize':8,

'line_ind_alpha':.05,
'line_ens_alpha':.9,

'hist_nbins':30,
'hist_kmin':0.,
'hist_kmax':30.,
'hist_pmin':0.,
'hist_pmax':1.,

'kde_bandwidth':.1,
'filter_data':False,
'filter_ACF':False,
'filter_ACF_width':1.,

'show_ens':True,
'show_ind':True,
'show_mean':False,
'show_exp1':False,
'show_exp2':False,
'show_exp3':False,
'show_exp4':False,
'show_hmm':True,
'show_zero':True,
'show_textbox':True,

'remove_viterbi':True

}

class obj(object): ## generic class to take anything you throw at it...
	def __init__(self,*args):
		self.args = args

def filter(x,pp):
	from scipy.ndimage import gaussian_filter1d
	try:
		if pp['filter_ACF']:
			return gaussian_filter1d(x,pp['filter_ACF_width'])
	except:
		pass
	return x

@nb.njit
def exp1fxn(t,c1,k1,ev):
	norm = ev + c1
	if ev/norm <= 0.01 or k1 > 1.: return t*0. + np.inf
	y = c1*np.exp(-k1*t) #+ b
	y[np.nonzero(t==0.)] += ev
	return y/norm

@nb.njit
def exp2fxn(t,c1,c2,k1,k2,ev):
	norm = ev + c1 + c2
	if ev/norm <= 0.01 or k1 > 1. or k2 > 1.: return t*0. + np.inf
	y = c1*np.exp(-k1*t) + c2*np.exp(-k2*t) #+ b
	y[np.nonzero(t==0.)] += ev
	return y/norm

@nb.njit
def exp3fxn(t,c1,c2,c3,k1,k2,k3,ev):
	norm = ev + c1 + c2 + c3
	if ev/norm <= 0.01 or k1 > 1. or k2 > 1. or k3 > 1.: return t*0. + np.inf
	y = c1*np.exp(-k1*t) + c2*np.exp(-k2*t) + c3*np.exp(-k3*t) #+ b
	y[np.nonzero(t==0.)] += ev
	return y/norm

@nb.njit
def exp4fxn(t,c1,c2,c3,c4,k1,k2,k3,k4,ev):
	norm = ev + c1 + c2 + c3 + c4
	if ev/norm <= 0.01 or k1 > 1. or k2 > 1. or k3 > 1. or k4 > 1.: return t*0. + np.inf
	y = c1*np.exp(-k1*t) + c2*np.exp(-k2*t) + c3*np.exp(-k3*t) + c4*np.exp(-k4*t) #+ b
	y[np.nonzero(t==0.)] += ev
	return y/norm

def lor1fxn(w,c1,k1,ev):
	norm = ev + c1
	# b = 0.
	# ev = 1.-c1
	if ev <= 0. or ev > 1.:# or k1 < 0.:# or c1 < 0:
		return w*0. + np.inf
	y = c1*S_exp(w,k1) + ev/np.sqrt(2.*np.pi)
	# y[np.nonzero(w==0.)] += b*np.sqrt(2.*np.pi)
	return y/norm

def lor2fxn(w,c1,c2,k1,k2,ev):
	# b = 1. - c1 - c2 - ev
	# ev = 1.-c1-c2
	norm = ev + c1 + c2
	if ev <= 0. or ev > 1.:# or k1 < 0. or k2 < 0.:# or c1 < 0 or c2 < 0:
		return w*0. + np.inf
	y = c1*S_exp(w,k1) + c2*S_exp(w,k2) + ev/np.sqrt(2.*np.pi)
	# y[np.nonzero(w==0.)] += b*np.sqrt(2.*np.pi)
	return y/norm

def lor3fxn(w,c1,c2,c3,k1,k2,k3,ev):
	# b = 1. - c1 - c2 - c3 - ev
	norm = ev + c1 + c2 + c3
	if ev <= 0. or ev > 1.:# or k1 < 0. or k2 < 0. or k3 < 0.:# or c1 < 0 or c2 < 0 or c3 < 0:
		return w*0. + np.inf
	y = c1*S_exp(w,k1) + c2*S_exp(w,k2) + c3*S_exp(w,k3) + ev/np.sqrt(2.*np.pi)
	# y[np.nonzero(w==0.)] += b*np.sqrt(2.*np.pi)
	return y/norm

def lor4fxn(w,c1,c2,c3,c4,k1,k2,k3,k4,ev):
	# b = 1. - c1 - c2 - c3 - ev
	norm = ev + c1 + c2 + c3 + c4
	if ev <= 0. or ev > 1.:# or k1 < 0. or k2 < 0. or k3 < 0. or k4 < 0.:# or c1 < 0 or c2 < 0 or c3 < 0 or c4 < 0:
		return w*0. + np.inf
	y = c1*S_exp(w,k1) + c2*S_exp(w,k2) + c3*S_exp(w,k3) + c4*S_exp(w,k4) + ev/np.sqrt(2.*np.pi)
	# y[np.nonzero(w==0.)] += b*np.sqrt(2.*np.pi)
	return y/norm

def fit_exp1(t,d):
	c = 1.-(d[0]-d[1])
	x0 = np.array((c,10./t.max(),1.-c))
	out = minimize(lambda x: np.sum(np.square(exp1fxn(t,*x)-d)),x0=x0,method='Nelder-Mead')
	print 1,out.success,out.x
	return out.x

def fit_exp2(t,d):
	c = 1.-(d[0]-d[1])
	x0 = np.array((c*.5,c*.5,30./t.max(),10./t.max(),1.-c))
	out = minimize(lambda x: np.sum(np.square(exp2fxn(t,*x)-d)),x0=x0,method='Nelder-Mead')
	print 2,out.success,out.x
	return out.x

def fit_exp3(t,d):
	c = 1.-(d[0]-d[1])
	x0 = np.array((c/3.,c/3.,c/3.,100./t.max(),30./t.max(),10./t.max(),1.-c))
	out = minimize(lambda x: np.sum(np.square(exp3fxn(t,*x)-d)),x0=x0,method='Nelder-Mead')
	print 3,out.success,out.x
	return out.x

def fit_exp4(t,d):
	c = 1.-(d[0]-d[1])
	x0 = np.array((c/4.,c/4.,c/4.,c/4.,100./t.max(),60./t.max(),30./t.max(),10./t.max(),1.-c))
	out = minimize(lambda x: np.sum(np.square(exp4fxn(t,*x)-d)),x0=x0,method='Nelder-Mead')
	print 4,out.success,out.x
	return out.x

def fit_lor1(w,d):
	c = 1.-d[-50].mean()
	x0 = np.array((c,w.max()/5.,1.-c))
	keep = w >= 0
	out = minimize(lambda x: np.sum(np.square(lor1fxn(w[keep],*x)-d[keep])),x0=x0,method='Nelder-Mead')
	print 1,out.success,out.x
	print out
	return out.x

def fit_lor2(w,d):
	c = 1.-d[-50].mean()
	x0 = np.array((c*.5,c*.5,w.max()/5.,w.max()/80.,1.-c))
	keep = w >= 0
	out = minimize(lambda x: np.sum(np.square(lor2fxn(w[keep],*x)-d[keep])),x0=x0,method='Nelder-Mead')
	print 2,out.success,out.x
	print out
	return out.x

def fit_lor3(w,d):
	c = 1.-d[-50].mean()
	x0 = np.array((c*.25,c*.5,c*.25,w.max()/5.,w.max()/8.,w.max()/11.,1.-c))
	keep = w >= 0
	out = minimize(lambda x: np.sum(np.square(lor3fxn(w[keep],*x)-d[keep])),x0=x0,method='Nelder-Mead')
	print 3,out.success,out.x
	print out
	return out.x

def fit_lor4(w,d):
	c = 1.-d[-50].mean()
	x0 = np.array((c*.2,c*.3,c*.3,c*.2,w.max()/5.,w.max()/8.,w.max()/11.,w.max()/15.,1.-c))
	keep = w >= 0
	out = minimize(lambda x: np.sum(np.square(lor4fxn(w[keep],*x)-d[keep])),x0=x0,method='Nelder-Mead')
	print 4,out.success,out.x
	print out
	return out.x

def kde(x,d,bw=None):
	from scipy.stats import gaussian_kde
	if bw is None:
		kernel = gaussian_kde(d)
	else:
		kernel = gaussian_kde(d,bw)
	y = kernel(x)
	return y

def power_spec(t,y):
	dt = t[1]-t[0]

	# tt = np.linspace(-t.max(),t.max(),t.size*2-1)
	# yy = np.zeros_like(tt)
	# yy[:t.size] = y[::-1]
	# yy[t.size:] = y[1:]
	# f = np.fft.fft(yy)*dt/np.sqrt(2.*np.pi)
	# w = np.fft.fftfreq(tt.size)*2.*np.pi/dt

	f = np.fft.fft(y)*dt/np.sqrt(2.*np.pi)
	w = np.fft.fftfreq(t.size)*2.*np.pi/dt
	# f /= f[0] ## normalize to zero frequency
	x = w.argsort()
	return w[x],np.abs(f)[x]

def S_exp(w,k):
	# analytic = k/(k*k+4.*np.pi*np.pi*w*w)
	# analytic *= k ## normalize to zero freq point
	# analytic = k/np.sqrt(2.*np.pi)/(k**2.+w**2.)
	analytic = np.sqrt(2./np.pi)*k/(k**2.+w**2.)
	return analytic

def setup(gui):
	recalcbutton = QPushButton("Recalculate")
	gui.popout_plots['plot_acorr'].ui.buttonbox.insertWidget(1,recalcbutton)
	recalcbutton.clicked.connect(lambda x: recalc(gui))

	gui.popout_plots['plot_acorr'].ui.combo_plot = QComboBox()
	gui.popout_plots['plot_acorr'].ui.combo_plot.addItems(['ACF','Power','Randomness'])
	gui.popout_plots['plot_acorr'].ui.buttonbox.insertWidget(2,gui.popout_plots['plot_acorr'].ui.combo_plot)
	gui.popout_plots['plot_acorr'].ui.combo_plot.setCurrentIndex(0)

	if not gui.data.d is None:
		recalc(gui)

def recalc(gui):
	popplot = gui.popout_plots['plot_acorr'].ui
	pp = popplot.prefs

	if gui.ncolors != 2:
		return

	from ..supporting.autocorr import ensemble_bayes_acorr,credible_interval,gen_mc_acf,gen_mc_acf_q
	from scipy.ndimage import gaussian_filter1d

	## get data
	popplot.fpb = gui.data.get_plot_data(pp['filter_data'])[0].copy()
	popplot.fpb[np.greater(popplot.fpb,1.25)] = 1.25
	popplot.fpb[np.less(popplot.fpb,-.25)] = -.25

	hr = gui.data.hmm_result
	if not hr is None and pp['remove_viterbi']:
		if hr.type == 'consensus vbfret':
			mu = hr.result.mu
			for i in range(popplot.fpb.shape[0]):
				v = hr.result.viterbi[i]
				pre = gui.data.pre_list[hr.ran[i]]
				popplot.fpb[i,pre:pre+v.size] -= mu[v]
		elif hr.type == 'vb' or hr.type == 'ml':
			for i in range(popplot.fpb.shape[0]):
				mu = hr.results[i].mu
				v = hr.results[i].viterbi
				pre = gui.data.pre_list[hr.ran[i]]
				popplot.fpb[i,pre:pre+v.size] -= mu[v]




	baseline = np.nanmean(popplot.fpb)

	t = np.arange(popplot.fpb.shape[1])

	#### Ensemble Data
	## posterior - m,k,a,b vs time
	## ci - credible intervals vs time
	## y - ACF vs time
	## t - time
	## fft - power spectrum
	## freq - power spectrum frequency axis
	## tc - correlation time
	## psp - lorentzian fits of PS to markov chain w/ normal emissions model

	popplot.ens = obj()
	popplot.ens.posterior = ensemble_bayes_acorr(popplot.fpb-baseline)

	norm = popplot.ens.posterior[0][0]
	popplot.ens.y = filter(popplot.ens.posterior[0],pp) / norm
	popplot.ens.t = t
	popplot.ens.ci = credible_interval(popplot.ens.posterior)
	for i in range(2):
		popplot.ens.ci[i] = filter(popplot.ens.ci[i],pp)/norm

	popplot.ens.freq,popplot.ens.fft = power_spec(popplot.ens.t,popplot.ens.y)

	popplot.ens.tc = np.sum(popplot.ens.y)
	# popplot.ens.psp1 = fit_lor1(popplot.ens.freq,popplot.ens.fft)
	# popplot.ens.psp2 = fit_lor2(popplot.ens.freq,popplot.ens.fft)
	# popplot.ens.psp3 = fit_lor3(popplot.ens.freq,popplot.ens.fft)
	# popplot.ens.psp4 = fit_lor4(popplot.ens.freq,popplot.ens.fft)
	popplot.ens.psp1 = fit_exp1(popplot.ens.t,popplot.ens.y)
	popplot.ens.psp2 = fit_exp2(popplot.ens.t,popplot.ens.y)
	popplot.ens.psp3 = fit_exp3(popplot.ens.t,popplot.ens.y)
	popplot.ens.psp4 = fit_exp4(popplot.ens.t,popplot.ens.y)

	#### Individual Data
	## y - list of ACF means
	## t - ACF time
	## k - array of single exp fit ACF rate constants
	## tc - correlation time
	## fft - array of power spectra
	## freq - power spectrum freqs
	## k_kde_x - rate constant axis
	## k_kde_y - probability of rate constant
	## exp_ps - analytical power spectrum for single exponential from 'k'

	popplot.ind = obj()
	popplot.ind.y = []
	for i in range(popplot.fpb.shape[0]):
		ff = popplot.fpb[i].reshape((1,popplot.fpb[i].size)) - baseline
		if not np.all(np.isnan(ff)):
			posterior = ensemble_bayes_acorr(ff)#-np.nanmean(ff))
			inorm = posterior[0][0]
			yyy = filter(posterior[0],pp)/inorm
			popplot.ind.y.append(yyy)
	popplot.ind.y = np.array(popplot.ind.y)
	popplot.ind.t = t

	popplot.ind.psp1 = []
	popplot.ind.fft = []
	popplot.ind.tc = []
	for i in range(popplot.ind.y.shape[0]):
		ft,f = power_spec(popplot.ind.t,popplot.ind.y[i])
		# popplot.ind.psp1.append(fit_lor1(ft[x],f[x]))
		popplot.ind.fft.append(f)
		tc = 1./np.sum(popplot.ind.y[i])
		if tc < 0:
			tc = 0
		popplot.ind.tc.append(tc)
	popplot.ind.freq = ft
	popplot.ind.fft = np.array(popplot.ind.fft)
	popplot.ind.psp1 = np.array(popplot.ind.psp1)
	popplot.ind.tc = np.array(popplot.ind.tc)

	try:
		ksx = np.isfinite(popplot.ind.k1[:,1])
		popplot.ind.k_kde_x = np.linspace(0,popplot.ind.psp[:,1][ksx].max()*1.2,1000)
		popplot.ind.k_kde_y = kde(popplot.ind.k_kde_x,popplot.ind.psp[:,1][ksx],pp['kde_bandwidth'])

		tcx = np.isfinite(popplot.ind.tc)
		popplot.ind.tc_kde_x = np.linspace(0,popplot.ind.tc[tcx].max()*1.2,1000)
		popplot.ind.tc_kde_y = kde(popplot.ind.tc_kde_x,popplot.ind.tc[tcx],pp['kde_bandwidth'])
	except:
		pass

	popplot.ind.y[np.bitwise_and((popplot.ind.y != 0), (np.roll(popplot.ind.y,-1,axis=1)-popplot.ind.y == 0.))] = np.nan

	#### HMM Data
	## t - ACF time
	## y - ACF
	## freq - Power spectrum frequency
	## fft - Power spectrum
	## k - single exponential fit rate constant
	## tc - correlation time
	## exp_ps - analytical power spectrum for single exponential from 'k'
	hr = gui.data.hmm_result
	popplot.hmm = None
	if not hr is None:
		popplot.hmm = obj()
		if hr.type == 'consensus vbfret':
			mu = hr.result.mu
			var = hr.result.var
			tmatrix = hr.result.tmstar
			ppi = hr.result.ppi
			popplot.hmm.t,popplot.hmm.y = gen_mc_acf(1.,popplot.ens.y.size,tmatrix,mu,var,ppi)
			popplot.hmm.freq,popplot.hmm.fft = power_spec(popplot.hmm.t,popplot.hmm.y)
			popplot.hmm.tc = np.sum(popplot.hmm.y)

		# elif hr.type == 'vb':
		# 	popplot.hmm.y = np.zeros_like(popplot.ens.y)
		# 	for i in range(popplot.fpb.shape[0]):
		# 		mu = hr.results[i].mu
		# 		var = hr.results[i].var
		# 		tmatrix = hr.results[i].tmstar
		# 		ppi = hr.results[i].ppi
		# 		t,y = gen_mc_acf(1.,popplot.ens.y.size,tmatrix,mu,var,ppi)
		# 		popplot.hmm.y += y/popplot.fpb.shape[0]
		# 	popplot.hmm.t = t
		# 	popplot.hmm.freq,popplot.hmm.fft = power_spec(popplot.hmm.t,popplot.hmm.y)
		# 	popplot.hmm.tc = np.sum(popplot.hmm.y)



def plot(gui):
	if gui.data.d is None:
		return
	popplot = gui.popout_plots['plot_acorr'].ui
	pp = popplot.prefs
	popplot.ax[0].cla()
	popplot.resize_fig()
	gui.app.processEvents()

	dpr = popplot.f.canvas.devicePixelRatio()

	method_index = gui.popout_plots['plot_acorr'].ui.combo_plot.currentIndex()

	#### Autocorrelation Function Plot
	if method_index == 0:
		plot_autocorrelation(gui,popplot,pp)

	#### Power Spectrum Plot
	elif method_index == 1:
		plot_powerspectrum(gui,popplot,pp)

	# ## histogram of ACF decay rates
	# elif method_index in [2,3]:
	# 	plot_histogram(gui,popplot,pp,method_index)
	elif method_index == 2:
		plot_randomness(gui,popplot,pp)

	# ####################################################
	# ####################################################

	fs = pp['label_fontsize']/dpr
	font = {
		'family': pp['font'],
		'size': fs,
		'va':'top'
	}

	if method_index == 0:
		popplot.ax[0].set_xlabel(pp['xlabel_text1'],fontdict=font)
		popplot.ax[0].set_ylabel(pp['ylabel_text1'],fontdict=font)
		if not pp['time_scale'] == 'log':
			popplot.ax[0].set_xticks(popplot.figure_out_ticks(pp['time_min'],pp['time_max'],pp['time_nticks']))
		popplot.ax[0].set_yticks(popplot.figure_out_ticks(pp['acorr_min'],pp['acorr_max'],pp['acorr_nticks']))
	elif method_index == 1:
		popplot.ax[0].set_xlabel(pp['xlabel_text2'],fontdict=font)
		popplot.ax[0].set_ylabel(pp['ylabel_text2'],fontdict=font)
	elif method_index == 2:
		popplot.ax[0].set_xlabel(r'Time (sec)',fontdict=font)
		popplot.ax[0].set_ylabel(r'Randomness, r',fontdict=font)
	# 	popplot.ax[0].set_ylabel(r'Probability',fontdict=font)
	# elif method_index == 3:
	# 	popplot.ax[0].set_xlabel(r'ACF inverse Correlation Time (sec$^{-1}$)',fontdict=font)
	# 	popplot.ax[0].set_ylabel(r'Probability',fontdict=font)
	# elif method_index == 4:
	# 	popplot.ax[0].set_xlabel(r'ACF Single Exponential Relaxation Rate (sec$^{-1}$)',fontdict=font)
	# 	popplot.ax[0].set_ylabel(r'ACF Correlation Time Relaxation Rate (sec$^{-1}$)',fontdict=font)

	popplot.ax[0].yaxis.set_label_coords(pp['ylabel_offset'], 0.5)
	popplot.ax[0].xaxis.set_label_coords(0.5, pp['xlabel_offset'])

	if pp['show_textbox']:
		bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=1./dpr)
		lstr = 'N = %d'%(popplot.fpb.shape[0])
		popplot.ax[0].annotate(lstr,xy=(pp['textbox_x'],pp['textbox_y']),xycoords='axes fraction', ha='right', color='k', bbox=bbox_props, fontsize=pp['textbox_fontsize']/dpr)

	fd = {'rotation':pp['xlabel_rotate'], 'ha':'center'}
	if fd['rotation'] != 0: fd['ha'] = 'right'
	popplot.ax[0].set_xticklabels(["{0:.{1}f}".format(x, pp['xlabel_decimals']) for x in popplot.ax[0].get_xticks()], fontdict=fd)
	popplot.ax[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: "{0:.{1}f}".format(x,pp['xlabel_decimals'])))

	fd = {'rotation':pp['ylabel_rotate']}
	popplot.ax[0].set_yticklabels(["{0:.{1}f}".format(y, pp['ylabel_decimals']) for y in popplot.ax[0].get_yticks()], fontdict=fd)
	popplot.ax[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: "{0:.{1}f}".format(x,pp['ylabel_decimals'])))


	popplot.f.canvas.draw()


def plot_autocorrelation(gui,popplot,pp):
	tau = pp['time_dt']

	if pp['show_zero']:
		popplot.ax[0].axhline(y=0.,color='k',alpha=.5,lw=1.)

	## Ensemble plots
	if pp['show_ens']:
		popplot.ax[0].fill_between(popplot.ens.t*tau, popplot.ens.ci[0], popplot.ens.ci[1], alpha=.3, color=pp['line_color'])
		popplot.ax[0].plot(popplot.ens.t*tau, popplot.ens.y, color=pp['line_color'], lw=1., alpha=pp['line_ens_alpha'])
	if pp['show_exp1']:
		popplot.ax[0].plot(popplot.ens.t*tau, exp1fxn(popplot.ens.t,*popplot.ens.psp1), color='red', lw=1., alpha=pp['line_ens_alpha'])
	if pp['show_exp2']:
		popplot.ax[0].plot(popplot.ens.t*tau, exp2fxn(popplot.ens.t,*popplot.ens.psp2), color='red', lw=1., alpha=pp['line_ens_alpha'])
	if pp['show_exp3']:
		popplot.ax[0].plot(popplot.ens.t*tau, exp3fxn(popplot.ens.t,*popplot.ens.psp3), color='red', lw=1., alpha=pp['line_ens_alpha'])
	if pp['show_exp4']:
		popplot.ax[0].plot(popplot.ens.t*tau, exp4fxn(popplot.ens.t,*popplot.ens.psp4), color='red', lw=1., alpha=pp['line_ens_alpha'])
	if pp['show_ind']:
		for i in range(popplot.ind.y.shape[0]):
			popplot.ax[0].plot(popplot.ind.t*tau, popplot.ind.y[i], color='k', alpha=pp['line_ind_alpha'])
	if pp['show_mean']:
		popplot.ax[0].plot(popplot.ind.t*tau, np.nanmean(popplot.ind.y,axis=0), color='orange', alpha=pp['line_ens_alpha'])

	popplot.ax[0].set_xscale('linear')
	if pp['time_scale'] == 'log':
		popplot.ax[0].set_xscale('log')
		if pp['time_min'] < pp['time_dt']:
			pp['time_min'] = pp['time_dt']
	popplot.ax[0].set_xlim(pp['time_min'],pp['time_max'])
	popplot.ax[0].set_ylim(pp['acorr_min'],pp['acorr_max'])

	if pp['show_hmm']:
		if not gui.data.hmm_result is None and not popplot.hmm is None:
			hr = gui.data.hmm_result
			if hr.type in ['consensus vbfret']:
				popplot.ax[0].plot(popplot.hmm.t*tau, popplot.hmm.y, color='g',alpha=pp['line_ens_alpha'])

def plot_powerspectrum(gui,popplot,pp):
	tau = pp['time_dt']
	f = popplot.ens.freq

	if pp['show_ens']:
		popplot.ax[0].semilogy(f/tau, popplot.ens.fft,lw=1.,color='b',alpha=pp['line_ens_alpha'])
	if pp['show_exp1']:
		popplot.ax[0].semilogy(f/tau,lor1fxn(f,*popplot.ens.psp1),color='red',lw=1,alpha=pp['line_ens_alpha'])
	if pp['show_exp2']:
		popplot.ax[0].semilogy(f/tau,lor2fxn(f,*popplot.ens.psp2),color='red',lw=1,alpha=pp['line_ens_alpha'])
	if pp['show_exp3']:
		popplot.ax[0].semilogy(f/tau,lor3fxn(f,*popplot.ens.psp3),color='red',lw=1,alpha=pp['line_ens_alpha'])
	if pp['show_exp4']:
		popplot.ax[0].semilogy(f/tau,lor4fxn(f,*popplot.ens.psp4),color='red',lw=1,alpha=pp['line_ens_alpha'])

	if pp['show_ind']:
		for i in range(popplot.ind.y.shape[0]):
			popplot.ax[0].semilogy(popplot.ind.freq/tau,np.abs(popplot.ind.fft[i]),color='k',alpha=pp['line_ind_alpha'],zorder=-2)
	if pp['show_mean']:
		q = np.nanmean(popplot.ind.y,axis=0)
		w,f = power_spec(popplot.ens.t,q)
		popplot.ax[0].semilogy(w/tau, f, color='orange', alpha=pp['line_ens_alpha'])

	popplot.ax[0].set_ylim(pp['power_min'],pp['power_max'])
	ft = popplot.ind.freq/tau
	popplot.ax[0].set_xlim(ft[ft>0].min(),ft[ft>0].max())
	popplot.ax[0].set_xscale('log')
	popplot.ax[0].set_yscale('log')

	if pp['show_hmm']:
		if not gui.data.hmm_result is None and not popplot.hmm is None:
			hr = gui.data.hmm_result
			if hr.type in ['consensus vbfret']:
				popplot.ax[0].plot(popplot.hmm.freq/tau, popplot.hmm.fft, color='g',alpha=pp['line_ens_alpha'])

def plot_histogram(gui,popplot,pp,method_index):
	tau = pp['time_dt']

	if method_index == 2:
		d = popplot.ind.psp1[:,1]/tau
		x = popplot.ind.k_kde_x / tau
		y = popplot.ind.k_kde_y
	elif method_index == 3:
		d = popplot.ind.tc/tau
		x = popplot.ind.tc_kde_x / tau
		y = popplot.ind.tc_kde_y
	print d.min(),np.mean(d),d.max()
	popplot.ax[0].hist(d,bins=pp['hist_nbins'],density=True,color='b',histtype='stepfilled',alpha=.6)
	popplot.ax[0].plot(x,y,'k',alpha=pp['line_ens_alpha'])
	popplot.ax[0].set_ylim(pp['hist_pmin'],pp['hist_pmax'])

def plot_randomness(gui,popplot,pp):
	tau = pp['time_dt']

	y = np.nanmean(popplot.fpb,axis=0)
	r = (np.nanmean(popplot.fpb**2.,axis=0)-y**2.)/y ## randomness parameter (Schnitzer/Block)
	t = tau * np.arange(y.size)
	popplot.ax[0].axhline(y=np.nanmean(r), color='k', lw=1., alpha=pp['line_ens_alpha'])
	popplot.ax[0].plot(t,r, color='blue', lw=1., alpha=pp['line_ens_alpha'])
	popplot.ax[0].set_xlim(t.min(),t.max())
	popplot.ax[0].set_ylim(-.2,1.2)
