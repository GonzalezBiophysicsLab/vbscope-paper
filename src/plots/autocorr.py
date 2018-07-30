import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PyQt5.QtWidgets import QPushButton, QComboBox

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
'show_textbox':True

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

def exp1fxn(t,k):
	if k <= 0:
		return t*0 + np.inf
	return np.exp(-k*t)

def exp2fxn(t,a1,a2,k1,k2):
	if k1 <= 0 or k2 <= 0 or k1 > 10. or k2 > 10.:
		return t*0 + np.inf
	return a1*np.exp(-k1*t) + a2*np.exp(-k2*t)

def exp3fxn(t,a1,a2,a3,k1,k2,k3):
	if k1 <= 0 or k2 <= 0 or k3 <= 0 or k1 > 10 or k2 > 10 or k3 > 10 :
		return t*0 + np.inf
	return a1*np.exp(-k1*t) + a2*np.exp(-k2*t) + a3*np.exp(-k3*t)

def exp4fxn(t,a1,a2,a3,a4,k1,k2,k3,k4):
	if k1 <= 0 or k2 <= 0 or k3 <= 0 or k4 <= 0 or k1 > 1 or k2 > 1 or k3 > 1 or k4 > 1:
		return t*0 + np.inf
	return a1*np.exp(-k1*t) + a2*np.exp(-k2*t) + a3*np.exp(-k3*t) + a4*np.exp(-k4*t)


def fit_exp1(t,d):
	k0 = 1./(t[np.argmax(d<.3679)]+1e-6)
	try:
		p,c = curve_fit(exp1fxn,t,d,p0=[k0],maxfev=1000)
		if not np.any(c == np.inf):
			return p
		else:
			return np.array([np.nan])
	except:
		return np.array([np.nan])


def fit_exp2(t,d):
	p = fit_exp1(t,d)
	x0 = [.5,.5,p[0]*10.,p[0]/10.]
	n = -1
	while n*-1 < d.size:
		try:
			p,c = curve_fit(exp2fxn,t[:n],d[:n],p0=x0,maxfev=1000)
			if not np.any(c == np.inf):
				break
		except:
			pass
		n -= 10
	return p
def fit_exp3(t,d):
	p = fit_exp2(t,d)
	if p[2] >= p[3]:
		k = np.copy(p[3])
		a = np.copy(p[1])
		p[3] = p[2]
		p[1] = p[0]
		p[0] = a
		p[2] = k
	x0 = [p[0]/2.,p[1]/2.,(p[0]+p[1])/2.,p[2]/2.,p[3]*2,(p[2]+p[3])/2.]
	# n = -1
	# while n*-1 < d.size:
	if 1:
		p = x0
		try:
	# if 1:
			p,c = curve_fit(exp3fxn,t[:-2],d[:-2],p0=x0,maxfev=1000)
			# if not np.any(c == np.inf):
				# break
		except:
			pass
		# n -= 10
	return p

def fit_exp4(t,d):
	p = fit_exp1(t,d)
	x0 = [1./4,1./4,1./4,1./4,p[0]*2.,p[0]*1.5,p[0]/1.5,p[0]/2.]
	n = -1
	while n*-1 < d.size:
		try:
			p,c = curve_fit(exp4fxn,t[:n],d[:n],p0=x0,maxfev=1000)
			if not np.any(c == np.inf):
				break
		except:
			pass
		n -= 10
	return p


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
	f = np.fft.fft(y)*dt
	w = np.fft.fftfreq(y.size)/dt
	# f /= f[0] ## normalize to zero frequency
	x = w.argsort()
	return x,w,np.abs(f)

def exponential_power_spec(w,k):
	analytic = np.sqrt(1./(k**2.+(2.*np.pi*w)**2.))
	# analytic *= k ## normalize to zero freq point
	return analytic

def setup(gui):

	recalcbutton = QPushButton("Recalculate")
	gui.popout_plots['plot_acorr'].ui.buttonbox.insertWidget(1,recalcbutton)
	recalcbutton.clicked.connect(lambda x: recalc(gui))

	gui.popout_plots['plot_acorr'].ui.combo_plot = QComboBox()
	gui.popout_plots['plot_acorr'].ui.combo_plot.addItems(['ACF','Power','k Hist','Tc Hist'])
	gui.popout_plots['plot_acorr'].ui.buttonbox.insertWidget(2,gui.popout_plots['plot_acorr'].ui.combo_plot)
	gui.popout_plots['plot_acorr'].ui.combo_plot.setCurrentIndex(0)

	recalc(gui)

def recalc(gui):
	popplot = gui.popout_plots['plot_acorr'].ui
	pp = popplot.prefs

	if gui.ncolors != 2:
		return

	from ..supporting.autocorr import ensemble_bayes_acorr,credible_interval,gen_acf
	from scipy.ndimage import gaussian_filter1d

	## get data
	# popplot.fpb = gui.data.get_plot_data(True)[0].copy() ## FRET from filtered intensities
	popplot.fpb = gui.data.get_plot_data(pp['filter_data'])[0].copy() ## FRET from unfiltered intensities
	popplot.fpb[popplot.fpb > 1.5] = 1.5
	popplot.fpb[popplot.fpb<-.5] = -.5
	baseline = np.nanmean(popplot.fpb)

	t = np.arange(popplot.fpb.shape[1])

	#### Ensemble Data
	## posterior - m,k,a,b vs time
	## ci - credible intervals vs time
	## y - ACF vs time
	## t - time
	## fft - power spectrum
	## freq - power spectrum frequency axis
	## k - single exponential fit rate constant
	## tc - correlation time
	## exp_ps - analytical power spectrum for single exponential from 'k'

	popplot.ens = obj()
	popplot.ens.posterior = ensemble_bayes_acorr(popplot.fpb-baseline)

	norm = popplot.ens.posterior[0][0]
	popplot.ens.y = filter(popplot.ens.posterior[0],pp) / norm
	popplot.ens.t = t
	popplot.ens.ci = credible_interval(popplot.ens.posterior)
	for i in range(2):
		popplot.ens.ci[i] = filter(popplot.ens.ci[i],pp)/norm
	####

	# popplot.ens.y -= baseline
	# norm = popplot.ens.y[0]
	# popplot.ens.y /= norm
	# popplot.ens.ci -= baseline
	# popplot.ens.ci /= norm

	x,w,f = power_spec(popplot.ens.t,popplot.ens.y)
	popplot.ens.fft = f[x]
	popplot.ens.freq = w[x]
	try:
		popplot.ens.k = fit_exp1(popplot.ens.t,popplot.ens.y)
		popplot.ens.k2 = fit_exp2(popplot.ens.t,popplot.ens.y)
		popplot.ens.k3 = fit_exp3(popplot.ens.t,popplot.ens.y)
		popplot.ens.k4 = fit_exp4(popplot.ens.t,popplot.ens.y)
	except:
		pass

	popplot.ens.tc = np.sum(popplot.ens.y)
	popplot.ens.exp_ps = exponential_power_spec(popplot.ens.freq,popplot.ens.k[0])
	try:
		tp = popplot.ens.k2
		if not np.any(np.isnan(tp[0])):
			popplot.ens.exp_ps2 = tp[0]*exponential_power_spec(popplot.ens.freq,tp[2]) + tp[1]*exponential_power_spec(popplot.ens.freq,tp[3])
		tp = popplot.ens.k3
		if not np.any(np.isnan(tp[0])):
			popplot.ens.exp_ps3 = tp[0]*exponential_power_spec(popplot.ens.freq,tp[3]) + tp[1]*exponential_power_spec(popplot.ens.freq,tp[4]) + tp[2]*exponential_power_spec(popplot.ens.freq,tp[5])
		tp = popplot.ens.k4
		if not np.any(np.isnan(tp[0])):
			popplot.ens.exp_ps4 = tp[0]*exponential_power_spec(popplot.ens.freq,tp[4]) + tp[1]*exponential_power_spec(popplot.ens.freq,tp[5]) + tp[2]*exponential_power_spec(popplot.ens.freq,tp[6]) + tp[3]*exponential_power_spec(popplot.ens.freq,tp[7])
	except:
		pass

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
		posterior = ensemble_bayes_acorr(ff-np.nanmean(ff))
		inorm = posterior[0][0]
		yyy = filter(posterior[0],pp)/inorm
		popplot.ind.y.append(yyy)
		# popplot.ind.y.append(filter(posterior[0],pp))
		# popplot.ind.y.append(filter(posterior[0]/posterior[0][0],pp))
	popplot.ind.y = np.array(popplot.ind.y)
	popplot.ind.t = t

	popplot.ind.k = []
	popplot.ind.k2 = []
	popplot.ind.fft = []
	popplot.ind.exp_ps = []
	popplot.ind.exp_ps2 = []
	popplot.ind.tc = []
	for i in range(popplot.ind.y.shape[0]):
		popplot.ind.k.append(fit_exp1(t,popplot.ind.y[i]))
		# popplot.ind.k2.append(fit_exp2(t,popplot.ind.y[i]))
		x,ft,f = power_spec(popplot.ind.t,popplot.ind.y[i])
		# popplot.ind.exp_ps.append(exponential_power_spec(ft,popplot.ind.k[i][0])
		# tp = popplot.ind.k2[i]
		# popplot.ind.exp_ps2.append(tp[0]*exponential_power_spec(ft,tp[2])+tp[1]*exponential_power_spec(ft,tp[3]))
		popplot.ind.fft.append(f)
		tc = 1./np.sum(popplot.ind.y[i])
		if tc < 0:
			tc = 0
		popplot.ind.tc.append(tc)
	popplot.ind.freq = ft[x]
	popplot.ind.fft = np.array(popplot.ind.fft)[:,x]
	popplot.ind.k = np.array(popplot.ind.k)
	# popplot.ind.k2 = np.array(popplot.ind.k2)
	popplot.ind.tc = np.array(popplot.ind.tc)
	# popplot.ind.exp_ps = np.array(popplot.ind.exp_ps)[:,x]
	# popplot.ind.exp_ps2 = np.array(popplot.ind.exp_ps2)[:,x]

	try:
		ksx = np.isfinite(popplot.ind.k)
		popplot.ind.k_kde_x = np.linspace(0,popplot.ind.k[ksx].max()*1.2,1000)
		popplot.ind.k_kde_y = kde(popplot.ind.k_kde_x,popplot.ind.k[ksx],pp['kde_bandwidth'])

		tcx = np.isfinite(popplot.ind.tc)
		popplot.ind.tc_kde_x = np.linspace(0,popplot.ind.tc[tcx].max()*1.2,1000)
		popplot.ind.tc_kde_y = kde(popplot.ind.tc_kde_x,popplot.ind.tc[tcx],pp['kde_bandwidth'])
	except:
		pass

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
			var = hr.result.var
			popplot.hmm.t,popplot.hmm.y = gen_acf(1.,popplot.ens.y.size,tmatrix,mu,var,ppi)
			# popplot.hmm.y -= b
			# popplot.hmm.y /= norm

			x,w,f = power_spec(popplot.hmm.t,popplot.hmm.y)
			popplot.hmm.fft = f[x]
			popplot.hmm.freq = w[x]

			popplot.hmm.k = fit_exp1(popplot.hmm.t,popplot.hmm.y)
			popplot.hmm.k2 = fit_exp2(popplot.hmm.t,popplot.hmm.y)
			popplot.hmm.k3 = fit_exp3(popplot.hmm.t,popplot.hmm.y)
			popplot.hmm.k4 = fit_exp4(popplot.hmm.t,popplot.hmm.y)
			popplot.hmm.tc = np.sum(popplot.hmm.y)
			popplot.hmm.exp_ps = exponential_power_spec(popplot.hmm.freq,popplot.hmm.k[0])
			tp = popplot.hmm.k2
			popplot.hmm.exp_ps2 =  tp[0]*exponential_power_spec(popplot.hmm.freq,tp[2])+tp[1]*exponential_power_spec(popplot.hmm.freq,tp[3])
		elif hr.type == 'vb':
			pass
			# popplot.hmm.y = []
			# popplot.hmm.fft = []
			# popplot.hmm.exp_ps = []
			# popplot.hmm.k = []
			# for i in range(popplot.fpb.shape[0]):
			# 	mu = hr.results[i].mu
			# 	var = hr.results[i].var
			# 	tmatrix = hr.results[i].tmstar
			# 	ppi = hr.results[i].ppi
			# 	popplot.hmm.t,y = gen_acf(1.,popplot.ens.y.size,tmatrix,mu,ppi)
			# 	y[~np.isfinite(y)] = 0.
			# 	popplot.hmm.y.append(y)
			# 	x,w,f = power_spec(popplot.hmm.t,popplot.hmm.y[i])
			# 	popplot.hmm.fft.append(f[x])
			# 	popplot.hmm.freq = w[x]
			# 	popplot.hmm.k.append(fit_exp1(popplot.hmm.t,popplot.hmm.y[i]))
			# 	popplot.hmm.exp_ps.append(exponential_power_spec(popplot.hmm.freq,popplot.hmm.k[i]))
			# popplot.hmm.y = np.array(popplot.hmm.y)
			# popplot.hmm.fft = np.array(popplot.hmm.fft)
			# popplot.hmm.exp_ps = np.array(popplot.hmm.exp_ps)
			# popplot.hmm.k = np.array(popplot.hmm.k)


def plot(gui):
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

	## histogram of ACF decay rates
	elif method_index in [2,3]:
		plot_histogram(gui,popplot,pp,method_index)

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
		popplot.ax[0].set_xlabel(r'ACF Relaxation Rate (sec$^{-1}$)',fontdict=font)
		popplot.ax[0].set_ylabel(r'Probability',fontdict=font)
	elif method_index == 3:
		popplot.ax[0].set_xlabel(r'ACF inverse Correlation Time (sec$^{-1}$)',fontdict=font)
		popplot.ax[0].set_ylabel(r'Probability',fontdict=font)
	elif method_index == 4:
		popplot.ax[0].set_xlabel(r'ACF Single Exponential Relaxation Rate (sec$^{-1}$)',fontdict=font)
		popplot.ax[0].set_ylabel(r'ACF Correlation Time Relaxation Rate (sec$^{-1}$)',fontdict=font)

	popplot.ax[0].yaxis.set_label_coords(pp['ylabel_offset'], 0.5)
	popplot.ax[0].xaxis.set_label_coords(0.5, pp['xlabel_offset'])

	if pp['show_textbox']:
		bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=1./dpr)
		lstr = 'N = %d'%(popplot.fpb.shape[0])
		popplot.ax[0].annotate(lstr,xy=(pp['textbox_x'],pp['textbox_y']),xycoords='axes fraction', ha='right', color='k', bbox=bbox_props, fontsize=pp['textbox_fontsize']/dpr)

	fd = {'rotation':pp['xlabel_rotate'], 'ha':'center'}
	if fd['rotation'] != 0: fd['ha'] = 'right'
	popplot.ax[0].set_xticklabels(["{0:.{1}f}".format(x, pp['xlabel_decimals']) for x in popplot.ax[0].get_xticks()], fontdict=fd)

	fd = {'rotation':pp['ylabel_rotate']}
	popplot.ax[0].set_yticklabels(["{0:.{1}f}".format(y, pp['ylabel_decimals']) for y in popplot.ax[0].get_yticks()], fontdict=fd)

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
		popplot.ax[0].plot(popplot.ens.t*tau, exp1fxn(popplot.ens.t,*popplot.ens.k), color='red', lw=1., alpha=pp['line_ens_alpha'])
	if pp['show_exp2']:
		popplot.ax[0].plot(popplot.ens.t*tau, exp2fxn(popplot.ens.t,*popplot.ens.k2), color='red', lw=1., alpha=pp['line_ens_alpha'])
	if pp['show_exp3']:
		popplot.ax[0].plot(popplot.ens.t*tau, exp3fxn(popplot.ens.t,*popplot.ens.k3), color='red', lw=1., alpha=pp['line_ens_alpha'])
	if pp['show_exp4']:
		popplot.ax[0].plot(popplot.ens.t*tau, exp4fxn(popplot.ens.t,*popplot.ens.k4), color='red', lw=1., alpha=pp['line_ens_alpha'])

	if pp['show_ind']:
		for i in range(popplot.ind.y.shape[0]):
			popplot.ax[0].plot(popplot.ind.t*tau, popplot.ind.y[i], color='k', alpha=pp['line_ind_alpha'])
	if pp['show_mean']:
		popplot.ax[0].plot(popplot.ind.t*tau, np.mean(popplot.ind.y,axis=0), color='orange', alpha=pp['line_ens_alpha'])

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
			if hr.type == 'consensus vbfret':
				popplot.ax[0].plot(popplot.hmm.t*tau, popplot.hmm.y, color='g',alpha=pp['line_ens_alpha'])

def plot_powerspectrum(gui,popplot,pp):
	tau = pp['time_dt']

	if pp['show_ens']:
		popplot.ax[0].semilogy(popplot.ens.freq/tau, popplot.ens.fft,lw=1.,color='b',alpha=pp['line_ens_alpha'])
	if pp['show_exp1']:
		popplot.ax[0].semilogy(popplot.ens.freq/tau, popplot.ens.exp_ps,lw=1.,color='red',alpha=pp['line_ens_alpha'])
	if pp['show_exp2']:
		popplot.ax[0].semilogy(popplot.ens.freq/tau, popplot.ens.exp_ps2,lw=1.,color='red',alpha=pp['line_ens_alpha'])
	if pp['show_exp3']:
		popplot.ax[0].semilogy(popplot.ens.freq/tau, popplot.ens.exp_ps3,lw=1.,color='red',alpha=pp['line_ens_alpha'])
	if pp['show_exp4']:
		popplot.ax[0].semilogy(popplot.ens.freq/tau, popplot.ens.exp_ps4,lw=1.,color='red',alpha=pp['line_ens_alpha'])

	if pp['show_ind']:
		for i in range(popplot.ind.y.shape[0]):
			popplot.ax[0].semilogy(popplot.ind.freq/tau,np.abs(popplot.ind.fft[i]),color='k',alpha=pp['line_ind_alpha'],zorder=-2)
	if pp['show_mean']:
		q = np.mean(popplot.ind.y,axis=0)
		x,w,f = power_spec(popplot.ens.t,q)
		popplot.ax[0].semilogy(w[x]/tau, f[x], color='orange', alpha=pp['line_ens_alpha'])

	popplot.ax[0].set_ylim(pp['power_min'],pp['power_max'])
	ft = popplot.ind.freq/tau
	popplot.ax[0].set_xlim(ft[ft>0].min(),ft[ft>0].max())
	popplot.ax[0].set_xscale('log')
	popplot.ax[0].set_yscale('log')

	if pp['show_hmm']:
		if not gui.data.hmm_result is None and not popplot.hmm is None:
			hr = gui.data.hmm_result
			if hr.type == 'consensus vbfret':
				popplot.ax[0].plot(popplot.hmm.freq/tau, popplot.hmm.fft, color='g',alpha=pp['line_ens_alpha'])

def plot_histogram(gui,popplot,pp,method_index):
	tau = pp['time_dt']

	if method_index == 2:
		d = popplot.ind.k/tau
		x = popplot.ind.k_kde_x / tau
		y = popplot.ind.k_kde_y
	elif method_index == 3:
		d = popplot.ind.tc/tau
		x = popplot.ind.tc_kde_x / tau
		y = popplot.ind.tc_kde_y
	print d.min(),np.mean(d),d.max()
	popplot.ax[0].hist(d,bins=pp['hist_nbins'],density=True,color='b',histtype='stepfilled',alpha=.6)
	popplot.ax[0].plot(x,y,'k',alpha=pp['line_ens_alpha'])
	# popplot.ax[0].set_xlim(pp['hist_kmin'],pp['hist_kmax'])
	popplot.ax[0].set_ylim(pp['hist_pmin'],pp['hist_pmax'])

	# elif method_index == 4:
	# 	d1 = popplot.ind.k/tau
	# 	d2 = popplot.ind.tc/tau
	# 	popplot.ax[0].plot(d1,d2,'o',alpha=.3,color='b')
	#
	# 	if not gui.data.hmm_result is None and not popplot.hmm is None:
	# 		hr = gui.data.hmm_result
	# 		# if hr.type == 'vb':
	# 		# 	kobs = popplot.ind.k*tau
	# 		# 	khmm = popplot.hmm.k*tau
	# 		# 	popplot.ax[0].plot(kobs,khmm,'o',alpha=.3,color='b')
