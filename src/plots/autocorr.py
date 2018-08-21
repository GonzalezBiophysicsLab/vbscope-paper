import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit,minimize
from PyQt5.QtWidgets import QPushButton, QComboBox
from matplotlib import ticker
import numba as nb
from math import lgamma,gamma
from scipy.special import gammaln

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

'power_nticks':6,
'power_min':.1,
'power_max':100.0,

'line_color':'blue',
'line_linewidth':1,
'line_alpha':0.9,

'fill_alpha':0.3,

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
'show_tc':True,
'show_stretch':True,
'show_hmm':True,
'show_zero':True,
'show_textbox':True,
'remove_viterbi':True,

'tc_cut':-1,

'beta_showens':False,
'beta_showmean':True,
'beta_nbins':41,
'tc_max':-1.,
'tc_min':-1.,
'tc_showgauss':True,
'tc_showkde':True,
'tc_nbins':41,
'tc_showens':False,
'tc_showmean':True,
'tc_fit_ymin':0.1,
'tc_ymax':0.5,
'tc_fitcut':3.,

'acorr_ind':0,

}

class obj(object): ## generic class to take anything you throw at it...
	def __init__(self,*args):
		self.args = args

def setup(gui):
	recalcbutton = QPushButton("Recalculate")
	gui.popout_plots['plot_acorr'].ui.buttonbox.insertWidget(1,recalcbutton)
	recalcbutton.clicked.connect(lambda x: recalc(gui))

	gui.popout_plots['plot_acorr'].ui.combo_plot = QComboBox()
	gui.popout_plots['plot_acorr'].ui.combo_plot.addItems(['ACF','Power','Mean','t_c','beta','tc v b','ind acf'])
	gui.popout_plots['plot_acorr'].ui.buttonbox.insertWidget(2,gui.popout_plots['plot_acorr'].ui.combo_plot)
	gui.popout_plots['plot_acorr'].ui.combo_plot.setCurrentIndex(0)

	gui.popout_plots['plot_acorr'].ui.nmol = 0

	pp = gui.popout_plots['plot_acorr'].ui.prefs
	clear_memory(gui)
	pp.commands['add to memory'] = lambda: add_to_memory(gui)
	pp.commands['clear memory'] = lambda: clear_memory(gui)
	pp.commands['cull short tc'] = lambda: cull_traces(gui)
	pp.update_commands()

	gui.popout_plots['plot_acorr'].ui.filter = None
	if not gui.data.d is None:
		recalc(gui)

def add_to_memory(gui):
	popplot = gui.popout_plots['plot_acorr'].ui
	popplot.prefs.memory.append([popplot.ens.t,popplot.ens.y,None,popplot.ens.freq,popplot.ens.fft,popplot.prefs['line_color']])

def clear_memory(gui):
	gui.popout_plots['plot_acorr'].ui.prefs.memory = []

def cull_traces(gui):
	if gui.popout_plots['plot_acorr'].ui.filter is None:
		gui.popout_plots['plot_acorr'].ui.filter = np.nonzero(np.bitwise_or(gui.popout_plots['plot_acorr'].ui.ind.tfit >= gui.popout_plots['plot_acorr'].ui.prefs['tc_fitcut'],~np.isnan(gui.popout_plots['plot_acorr'].ui.ind.tc)))[0]
		recalc(gui)

#############################################################################
@nb.vectorize
def vgamma(x):
	return gamma(x)

def filter(x,pp):
	from scipy.ndimage import gaussian_filter1d
	try:
		if pp['filter_ACF']:
			return gaussian_filter1d(x,pp['filter_ACF_width'])
	except:
		pass
	return x

def kde(x,d,bw=None):
	from scipy.stats import gaussian_kde
	if bw is None:
		kernel = gaussian_kde(d)
	else:
		kernel = gaussian_kde(d,bw)
	y = kernel(x)
	return y

#### Following Kaufman Lab - Mackowiak JCP 2009
@nb.njit
def stretched_exp(t,k,t0,b):
	if  b < 0 or t0 <= 0:
		return t*0 + np.inf
	# if t0 >= t[1]:
	else:
		return k*np.exp(-(t/t0)**b)
	# else:
		# q = np.zeros_like(t+k)
	# 	q[0] = 1.
		# return q
@nb.njit
def reg_exp(t,k,t0):
	if t0 <= 0:
		return t*0 + np.inf
	else:
		return k*np.exp(-(t/t0))

@nb.njit
def minfxn_stretch(t,y,x):
	k,t0,b = x
	f = stretched_exp(t[0],k,t0,b)
	tc = t0/b*vgamma(1./b)
	if b < 0. or f > 2.0 or f < 0.0 or t0 >= y.size*2. or tc > y.size or tc < 1.:
		return np.inf
	return np.sum(np.square(stretched_exp(t,k,t0,b) - y))

@nb.njit
def minfxn_reg(t,y,x):
	k,t0 = x
	f = reg_exp(t[0],k,t0)
	if f > 2.0 or f < 0.0 or t0 >= y.size*2. or t0 < 1.:
		return np.inf
	return np.sum(np.square(reg_exp(t,k,t0) - y))

# def line(t,c,tf):
# 	return c*(1. - t/tf)

def fit_acf(t,y,ymin = 0.05):
	yy = y<ymin
	if np.any(yy > 0):
		cutoff = np.argmax(yy)
		if cutoff > y.size:
			cutoff = -1
	else:
		cutoff = -1
	start = 1
	dt = t[1]-t[0]

	if y[start:cutoff].size >3:
		#### streched exponential
		m = np.max((dt,(y[start:cutoff]*t[start:cutoff]).sum()/y[start:cutoff].sum()))
		x0 = np.array((y[start],m,1.))
		out = minimize(lambda x: minfxn_stretch(t[start:cutoff],y[start:cutoff],x),x0,method='Nelder-Mead',options={'maxiter':1000})
		if out.success:
			return out.x
		#### exponential
		out = minimize(lambda x: minfxn_reg(t[start:cutoff],y[start:cutoff],x),x0[:-1],method='Nelder-Mead',options={'maxiter':1000})
		if out.success:
			return np.append(out.x,1.)

	# 	p,c = curve_fit(stretched_exp,t[start:cutoff],y[start:cutoff],p0=x0,maxfev=1000)
	# 	if np.all(np.isfinite(c)):
	# 		# if p[1] < (cutoff-start) and p[1] > 3:
	# 			return p
	# 	# x0 = x0[:-1]
	# 	# p,c = curve_fit(reg_exp,t[start:cutoff],y[start:cutoff],p0=x0,maxfev=1000)
	# 	# if np.all(np.isfinite(c)):
	# 	# 	# if p[1] < (cutoff-start) and p[1] > 3:
	# 	# 		return np.array((p[0],p[1],1.))
	#
	#
	#
	# 		else: ## linear fit
	# 			start = 1
	# 			m = (y[cutoff] - y[start])/(t[cutoff]-t[start])
	# 			c = y[start]-m*t[start]
	# 			x0 = np.array((c,-m/c))
	# 			p,c = curve_fit(line,t[start:cutoff],y[start:cutoff],p0=x0,maxfev=1000)
	# 			if np.all(np.isfinite(c)):
	# 				if p[1] < cutoff:
	# 					r = np.array((p[0],p[1],np.nan))
	# 					return r
	# except:
	# 	pass
	#
	#
	# 	if calc_tc(r) > (cutoff-start):
	# 		r = np.array((1.,1.,np.nan))
	# 	return r
	#
	out = np.array((1.,1.,np.nan))
	return out

def calc_tc(p):
	if p.ndim == 1:
		k,t,b = p
		if np.isnan(b):
			if t < 1: return 1.
			return t
		else:
			return t/b*vgamma(1./b)
	elif p.ndim == 2.:
		k,t,b = p
		out = np.zeros_like(k)
		xline = np.nonzero(np.isnan(b))[0]
		x = np.nonzero(np.isfinite(b))[0]
		out[xline] = t[xline]
		out[x] = (t[x]/b[x])*vgamma(1./b[x])
		# out[out < 1.] = 1.
		return out

def power_spec(t,y):
	dt = t[1]-t[0]
	f = np.fft.fft(y)*dt/np.sqrt(2.*np.pi)
	w = np.fft.fftfreq(t.size)*2.*np.pi/dt
	# f /= f[0] ## normalize to zero frequency
	x = w.argsort()
	return w[x],np.abs(f)[x]

def S_exp(w,k):
	analytic = np.sqrt(2./np.pi)*k/(k**2.+w**2.)
	return analytic

#############################################################################

def recalc(gui):
	popplot = gui.popout_plots['plot_acorr'].ui
	pp = popplot.prefs

	if gui.ncolors != 2:
		return

	from ..supporting.autocorr import acf_estimator, gen_mc_acf

	## get data
	popplot.fpb = gui.data.get_plot_data(pp['filter_data'])[0].copy()
	popplot.fpb[np.greater(popplot.fpb,1.5)] = .5 ### won't skew ACFs
	popplot.fpb[np.less(popplot.fpb,-.5)] = .5 ### won't skew ACFs

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

	if not popplot.filter is None:
		if len(popplot.filter) > 0:
			popplot.fpb = popplot.fpb[popplot.filter]
			popplot.filter = None

	baseline = np.nanmean(popplot.fpb)
	t = np.arange(popplot.fpb.shape[1])

	#### Ensemble Data
	## y - ACF vs time
	## t - time
	## fft - power spectrum
	## freq - power spectrum frequency axis
	## tc - correlation time
	## beta - stretched parameter

	popplot.ens = obj()
	popplot.ens.y = filter(acf_estimator(popplot.fpb),pp)
	popplot.ens.y /= popplot.ens.y[0] ## filtering can mess it up a little, therefore renormalize
	popplot.ens.t = t

	popplot.ens.freq,popplot.ens.fft = power_spec(popplot.ens.t,popplot.ens.y)

	popplot.ens.stretch_params = fit_acf(popplot.ens.t,popplot.ens.y,pp['tc_fit_ymin'])
	popplot.ens.tfit = popplot.ens.stretch_params[1]
	popplot.ens.tc = calc_tc(popplot.ens.stretch_params)
	popplot.ens.beta = popplot.ens.stretch_params[2]

	#### Individual Data
	## y - list of ACF means
	## t - ACF time
	## tc - correlation time
	## beta - stretched exponent
	## fft - array of power spectra
	## freq - power spectrum freqs

	popplot.ind = obj()
	popplot.ind.y = []
	for i in range(popplot.fpb.shape[0]):
		ff = popplot.fpb[i].reshape((1,popplot.fpb.shape[1]))
		if not np.all(np.isnan(ff)):
			popplot.ind.y.append(filter(acf_estimator(ff),pp))
	popplot.ind.y = np.array(popplot.ind.y)
	popplot.ind.y /= popplot.ind.y[:,0][:,None]
	popplot.ind.t = t

	popplot.ind.fft = []
	popplot.ind.tfit = []
	popplot.ind.tc = []
	popplot.ind.beta = []
	for i in range(popplot.ind.y.shape[0]):
		ft,f = power_spec(popplot.ind.t,popplot.ind.y[i])
		popplot.ind.fft.append(f)
		stretch_params = fit_acf(popplot.ind.t,popplot.ind.y[i],pp['tc_fit_ymin'])
		popplot.ind.tfit.append(stretch_params[1])
		popplot.ind.tc.append(calc_tc(stretch_params))
		popplot.ind.beta.append(stretch_params[2])
	popplot.ind.freq = ft

	popplot.ind.fft = np.array(popplot.ind.fft)
	popplot.ind.tc = np.array(popplot.ind.tc)
	popplot.ind.tfit = np.array(popplot.ind.tfit)
	popplot.ind.beta = np.array(popplot.ind.beta)

	popplot.ind.y[np.bitwise_and((popplot.ind.y != 0), (np.roll(popplot.ind.y,-1,axis=1)-popplot.ind.y == 0.))] = np.nan

	#### HMM Data
	## t - ACF time
	## y - ACF
	## freq - Power spectrum frequency
	## fft - Power spectrum
	## tc - correlation time
	## beta - stretched exponent
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
			popplot.hmm.stretch_params = fit_acf(popplot.hmm.t,popplot.hmm.y,pp['tc_fit_ymin'])
			popplot.hmm.tc = calc_tc(popplot.hmm.stretch_params)
			popplot.hmm.tfit = popplot.hmm.stretch_params[1]
			popplot.hmm.beta = popplot.hmm.stretch_params[2]
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

	elif method_index == 2:
		plot_mean(gui,popplot,pp)
	elif method_index == 3:
		plot_tc(gui,popplot,pp)
	elif method_index == 4:
		plot_beta(gui,popplot,pp)
	elif method_index == 5:
		plot_scatter(gui,popplot,pp)
	elif method_index == 6:
		plot_indacf(gui,popplot,pp)

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
		popplot.ax[0].set_ylabel(r'Mean',fontdict=font)
	elif method_index == 3:
		popplot.ax[0].set_xlabel(r'$ln(t_c)$',fontdict=font)
		popplot.ax[0].set_ylabel('Probability',fontdict=font)
	elif method_index == 4:
		popplot.ax[0].set_xlabel(r'$\beta$',fontdict=font)
		popplot.ax[0].set_ylabel('Counts',fontdict=font)
	elif method_index == 5:
		popplot.ax[0].set_ylabel(r'$t_c$',fontdict=font)
		popplot.ax[0].set_xlabel(r'$\beta$',fontdict=font)


	popplot.ax[0].yaxis.set_label_coords(pp['ylabel_offset'], 0.5)
	popplot.ax[0].xaxis.set_label_coords(0.5, pp['xlabel_offset'])

	if pp['show_textbox']:
		bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=1./dpr)
		lstr = r'N = %d'%(popplot.nmol)
		# print popplot.ens.tc*pp['time_dt'],np.median(popplot.ind.tc)*pp['time_dt']

		if pp['show_tc']:
			lstr += r', $t_c$=%.2f sec'%(np.around(popplot.ens.tc*pp['time_dt'],2))
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
		for mm in pp.memory:
			# popplot.ax[0].fill_between(mm[0]*tau, mm[2][0], mm[2][1], alpha=.3, color=mm[5],zorder=-2)
			popplot.ax[0].plot(mm[0]*tau, mm[1], color=mm[5], lw=1., alpha=pp['line_ens_alpha'],zorder=-1)

		# popplot.ax[0].fill_between(popplot.ens.t*tau, popplot.ens.ci[0], popplot.ens.ci[1], alpha=.3, color=pp['line_color'],zorder=0)
		popplot.ax[0].plot(popplot.ens.t*tau, popplot.ens.y, color=pp['line_color'], lw=1., alpha=pp['line_ens_alpha'],zorder=1)
	if pp['show_stretch']:
		popplot.ax[0].plot(popplot.ens.t*tau,stretched_exp(popplot.ens.t,*popplot.ens.stretch_params),color='r',lw=1,alpha=pp['line_ens_alpha'])

	if pp['show_ind']:
		for i in range(popplot.ind.y.shape[0]):
			popplot.ax[0].plot(popplot.ind.t*tau, popplot.ind.y[i], color='k', alpha=pp['line_ind_alpha'])
	if pp['show_mean']:
		popplot.ax[0].plot(popplot.ind.t*tau, np.nanmean(popplot.ind.y,axis=0), color='orange', alpha=pp['line_ens_alpha'])
		# popplot.ax[0].plot(popplot.ind.t*tau, np.median(popplot.ind.y,axis=0), color='orange', alpha=pp['line_ens_alpha'])
	popplot.nmol = popplot.fpb.shape[0]

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

				# dt = 0.025
				# mu = np.array((.1,.3,.6,.9))
				# noise = 0.11
				# var = np.zeros_like(mu) + noise**2.
				# rates = 10. * np.array(((0.,.005,.003,.001),(.001,0.,.002,.002),(.004,.003,.0,.004),(0.002,.003,.001,0.)))
				# from scipy.linalg import expm
				# q = rates.copy()
				# for i in range(q.shape[0]):
				# 	q[i,i] = - q[i].sum()
				# tmatrix = expm(q*dt)
				# tinf = 1000./np.abs(q).min()
				# ppi = expm(q*tinf)[0]
				# from ..supporting.autocorr import gen_mc_acf_q
				# t,y = gen_mc_acf_q(dt,popplot.ens.y.size,q,mu,var,ppi)
				# t /= dt
				# print y
				# print t[:10]
				# popplot.ax[0].plot(t,y/y[0],color='r',alpha=pp['line_ens_alpha'])

def plot_powerspectrum(gui,popplot,pp):
	tau = pp['time_dt']
	f = popplot.ens.freq

	if pp['show_ens']:
		for mm in pp.memory:
			popplot.ax[0].semilogy(mm[3]/tau, mm[4],lw=1.,color=mm[5],alpha=pp['line_ens_alpha'],zorder=0)

		popplot.ax[0].semilogy(f/tau, popplot.ens.fft,lw=1.,color=pp['line_color'],alpha=pp['line_ens_alpha'],zorder=1)
	if pp['show_stretch']:
		y = stretched_exp(popplot.ens.t,*popplot.ens.stretch_params)
		t = popplot.ens.t*tau
		w,f = power_spec(t,y)
		popplot.ax[0].semilogy(w, f, lw=1., color='red', alpha=pp['line_ens_alpha'], zorder=1)
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
	popplot.nmol = popplot.fpb.shape[0]

def plot_mean(gui,popplot,pp):
	tau = pp['time_dt']

	y = np.nanmean(popplot.fpb,axis=0)
	t = tau * np.arange(y.size)
	popplot.ax[0].plot(t,y, color='blue', lw=1., alpha=pp['line_ens_alpha'])
	popplot.ax[0].set_xlim(t.min(),t.max())
	popplot.ax[0].set_ylim(-.25,1.25)

	popplot.nmol = popplot.fpb.shape[0]

def plot_tc(gui,popplot,pp):
	tau = pp['time_dt']
	beta = popplot.ind.beta.copy()
	tc = popplot.ind.tc.copy()
	tf = popplot.ind.tfit.copy()
	tf_cut = pp['tc_fitcut']
	# x = tf >= tf_cut
	# tc = tc[x]
	y = np.log(tc*tau)
	y = y[np.isfinite(y)]
	ymin = np.log(pp['tc_fitcut']*tau)
	popplot.nmol = (y[y>ymin]).size

	if pp['tc_min'] > 0 :
		rmin = np.log(pp['tc_min'])
	else:
		rmin = np.log(tau/2.)
	if pp['tc_max'] > 0:
		rmax = np.log(pp['tc_max'])
	else:
		rmax = np.min((np.nanmax(y),np.log(popplot.ens.t.size/1.)))
	hy = popplot.ax[0].hist(y[y>ymin],bins=pp['tc_nbins'],range=(rmin,rmax),histtype='stepfilled',density=True)[0]
	# if pp['tc_showens']:
		# popplot.ax[0].axvline(x=np.log(popplot.ens.tc*tau),color='k')
	if pp['tc_showmean']:
		popplot.ax[0].axvline(x=np.nanmean(np.exp(y[y>ymin])),color='k',alpha=.9)

	if not gui.data.hmm_result is None and not popplot.hmm is None:
		hr = gui.data.hmm_result
		if hr.type in ['consensus vbfret']:
			popplot.ax[0].axvline(x=np.log(popplot.hmm.tc*tau),color='green',alpha=.9)

	# if pp['tc_showmean']:
	# 	ltc =  np.linspace(rmin,rmax,1000)
	# 	v = np.nanvar(np.log(tc*tau))
	# 	m = np.nanmean(np.log(tc*tau))
	# 	lp = (2.*np.pi*v)**-.5 * np.exp(-.5/v*(ltc-m)**2.)
	# 	popplot.ax[0].plot(ltc,lp,color='k',lw=1,alpha=.9)
	if pp['tc_showkde']:
		ltc =  np.linspace(rmin,rmax,1000)
		lp = kde(ltc,y[y>ymin])
		popplot.ax[0].plot(ltc,lp,color='k',lw=1,alpha=.9)
	popplot.ax[0].set_xlim(rmin,rmax)
	popplot.ax[0].set_ylim(0.,pp['tc_ymax'])

def plot_beta(gui,popplot,pp):
	beta = popplot.ind.beta.copy()
	tc = popplot.ind.tc.copy()
	tf = popplot.ind.tfit.copy()
	tf_cut = pp['tc_fitcut']
	x = tf >= tf_cut
	beta = beta[x]
	popplot.ax[0].hist(beta,bins=pp['beta_nbins'],range=(0,2),histtype='stepfilled')
	if pp['beta_showens']:
		popplot.ax[0].axvline(x=popplot.ens.beta,color='k')
	if pp['beta_showmean']:
		popplot.ax[0].axvline(x=np.nanmean(beta),color='k')
	popplot.ax[0].set_xlim(0,2.)
	popplot.nmol = beta.size

def plot_scatter(gui,popplot,pp):
	tau = pp['time_dt']
	beta = popplot.ind.beta.copy()
	tc = popplot.ind.tc.copy()
	tf = popplot.ind.tfit.copy()
	tf_cut = pp['tc_fitcut']
	x = (tf >= tf_cut)*	np.isfinite(beta)
	popplot.ax[0].loglog(beta[x],tc[x]*tau,'o',alpha=.5)
	x = (tf < tf_cut)
	popplot.ax[0].loglog(beta[x],tc[x]*tau,'o',alpha=.5,color='r')
	x = np.isnan(beta)
	popplot.ax[0].loglog(np.ones(int(x.sum())),tc[x]*tau,'o',alpha=.5,color='r')

	popplot.ax[0].set_xlim(.1,3.)
	popplot.ax[0].set_ylim(tau,tau*popplot.fpb.shape[1])
	bb = np.linspace(.1,3.,10000)
	p = np.array((np.ones(bb.size),np.zeros(bb.size)+tf_cut,bb))
	tt = calc_tc(p)*tau
	popplot.ax[0].plot(bb,tt,color='k',ls='--',lw=1.,alpha=.9)
	popplot.nmol = beta[x].size

	if not gui.data.hmm_result is None and not popplot.hmm is None:
		hr = gui.data.hmm_result
		if hr.type in ['consensus vbfret']:
			popplot.ax[0].axhline(y=popplot.hmm.tc*tau,color='green',alpha=.9)
			popplot.ax[0].axvline(x=popplot.hmm.beta,color='green',alpha=.9)

def plot_indacf(gui,popplot,pp):
	tau = pp['time_dt']

	ind = pp['acorr_ind']
	if ind < 0:
		pp['acorr_ind'] = 0
		plot_indacf(gui,popplot,pp)
		return
	elif ind >= popplot.ind.y.shape[0]:
		pp['acorr_ind'] = popplot.ind.y.shape[0] - 1
		plot_indacf(gui,popplot,pp)
		return
	popplot.ax[0].axhline(y=0,color='k',alpha=.5)

	popplot.ax[0].plot(popplot.ind.t*tau, popplot.ind.y[ind], color='k', alpha=pp['line_alpha'])
	stretch_params = fit_acf(popplot.ind.t,popplot.ind.y[ind],pp['tc_fit_ymin'])
	if np.isnan(stretch_params[2]):
		stretch_params[2] = 1.
	q = stretched_exp(popplot.ind.t,*stretch_params)
	popplot.ax[0].plot(popplot.ind.t*tau,q,color='r',alpha=pp['line_alpha'])

	popplot.ax[0].set_xscale('linear')
	if pp['time_scale'] == 'log':
		popplot.ax[0].set_xscale('log')
		if pp['time_min'] < pp['time_dt']:
			pp['time_min'] = pp['time_dt']
	popplot.ax[0].set_xlim(pp['time_min'],pp['time_max'])
	popplot.ax[0].set_ylim(pp['acorr_min'],pp['acorr_max'])
	k,t,b = stretch_params
	popplot.ax[0].set_title("%.3f: %.2f, %.3f, %.2f"%(calc_tc(stretch_params),k,t,b))
