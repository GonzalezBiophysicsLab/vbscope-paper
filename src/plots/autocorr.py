import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QPushButton, QComboBox
from matplotlib import ticker
# from ..supporting import autocorr as ac

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
'line_hmmcolor':'darkred',

'hist_color':'#0080FF',
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
'tc_fitcut':1.,

'acorr_ind':0,
'fit_biexp':False,

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
	pp.commands['write ind acf'] = lambda: write_ind_acf(gui)
	pp.update_commands()

	gui.popout_plots['plot_acorr'].ui.filter = None
	if not gui.data.d is None:
		recalc(gui)

def add_to_memory(gui):
	popplot = gui.popout_plots['plot_acorr'].ui
	popplot.prefs.memory.append([popplot.ens.t,popplot.ens.y,None,popplot.ens.freq,popplot.ens.fft,popplot.prefs['line_color']])

def clear_memory(gui):
	gui.popout_plots['plot_acorr'].ui.prefs.memory = []

def write_ind_acf(gui):
	from PyQt5.QtWidgets import QFileDialog

	popplot = gui.popout_plots['plot_acorr'].ui
	pp = popplot.prefs
	oname = QFileDialog.getSaveFileName(gui, 'Export ACF', '_acf.npy','*.npy')
	if oname[0] != "" and not oname[0] is None:
		o = np.array((popplot.ind.t,popplot.ind.y[pp['acorr_ind']]))
		np.save(oname[0],o)
		gui.log('Saved ACF %d'%(pp['acorr_ind']))


#############################################################################

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

#############################################################################

def recalc(gui):
	popplot = gui.popout_plots['plot_acorr'].ui
	pp = popplot.prefs

	if gui.ncolors != 2:
		return

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

	from ..supporting.autocorr import acf_estimator,power_spec,fit_acf

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

	popplot.ens.fit = fit_acf(popplot.ens.t,popplot.ens.y,pp['tc_fit_ymin'],False)
	popplot.ens.tc = popplot.ens.fit.calc_tc()
	if popplot.ens.fit.type == 'stretched exponential':
		popplot.ens.beta = popplot.ens.fit.params[2]
	else:
		popplot.ens.beta = np.nan

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
	popplot.ind.fit = []
	popplot.ind.tc = []
	popplot.ind.beta = []
	for i in range(popplot.ind.y.shape[0]):
		ft,f = power_spec(popplot.ind.t,popplot.ind.y[i])
		popplot.ind.fft.append(f)
		fit = fit_acf(popplot.ind.t,popplot.ind.y[i],pp['tc_fit_ymin'],pp['fit_biexp'])
		popplot.ind.fit.append(fit)
		popplot.ind.tc.append(fit.calc_tc())
		if fit.type == 'stretched exponential':
			popplot.ind.beta.append(fit.params[2])
		else:
			popplot.ind.beta.append(np.nan)
	popplot.ind.freq = ft

	popplot.ind.fft = np.array(popplot.ind.fft)
	popplot.ind.tc = np.array(popplot.ind.tc)
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
			from ..supporting.autocorr import gen_mc_acf

			mu = hr.result.mu
			var = hr.result.var
			tmatrix = hr.result.tmstar
			ppi = hr.result.ppi
			popplot.hmm.t,popplot.hmm.y = gen_mc_acf(1.,popplot.ens.y.size,tmatrix,mu,var,ppi)
			popplot.hmm.freq,popplot.hmm.fft = power_spec(popplot.hmm.t,popplot.hmm.y)
			popplot.hmm.fit = fit_acf(popplot.hmm.t,popplot.hmm.y,pp['tc_fit_ymin'],False)
			popplot.hmm.tc = popplot.hmm.fit.calc_tc()
			if popplot.hmm.fit.type == 'stretched exponential':
				popplot.hmm.beta = popplot.hmm.fit.params[2]
			else:
				popplot.hmm.beta = np.nan

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

	if method_index == 0:
		plot_autocorrelation(gui,popplot,pp)
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

	if pp['show_ind']:
		for i in range(popplot.ind.y.shape[0]):
			popplot.ax[0].plot(popplot.ind.t*tau, popplot.ind.y[i], color='k', alpha=pp['line_ind_alpha'])

	## Ensemble plots
	if pp['show_ens']:
		for mm in pp.memory:
			# popplot.ax[0].fill_between(mm[0]*tau, mm[2][0], mm[2][1], alpha=.3, color=mm[5],zorder=-2)
			popplot.ax[0].plot(mm[0]*tau, mm[1], color=mm[5], lw=1., alpha=pp['line_ens_alpha'],zorder=-1)

		# popplot.ax[0].fill_between(popplot.ens.t*tau, popplot.ens.ci[0], popplot.ens.ci[1], alpha=.3, color=pp['line_color'],zorder=0)
		popplot.ax[0].plot(popplot.ens.t*tau, popplot.ens.y, color=pp['line_color'], lw=1., alpha=pp['line_ens_alpha'],zorder=1)
	if pp['show_stretch']:
		popplot.ax[0].plot(popplot.ens.t*tau,popplot.ens.fit(popplot.ens.t),color='r',lw=1,alpha=pp['line_ens_alpha'])

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
				popplot.ax[0].plot(popplot.hmm.t*tau, popplot.hmm.y, color=pp['line_hmmcolor'],alpha=pp['line_ens_alpha'])

def plot_powerspectrum(gui,popplot,pp):
	from ..supporting.autocorr import power_spec
	tau = pp['time_dt']
	f = popplot.ens.freq

	if pp['show_ind']:
		for i in range(popplot.ind.y.shape[0]):
			popplot.ax[0].semilogy(popplot.ind.freq/tau,np.abs(popplot.ind.fft[i]),color='k',alpha=pp['line_ind_alpha'],zorder=-2)
	if pp['show_ens']:
		for mm in pp.memory:
			popplot.ax[0].semilogy(mm[3]/tau, mm[4],lw=1.,color=mm[5],alpha=pp['line_ens_alpha'],zorder=0)

		popplot.ax[0].semilogy(f/tau, popplot.ens.fft,lw=1.,color=pp['line_color'],alpha=pp['line_ens_alpha'],zorder=1)
	if pp['show_stretch']:

		y = popplot.ens.fit(popplot.ens.t)
		tt = popplot.ens.t
		ww,fft = power_spec(tt,y)
		popplot.ax[0].semilogy(ww/tau, fft, lw=1., color='red', alpha=pp['line_ens_alpha'], zorder=1)
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
				popplot.ax[0].plot(popplot.hmm.freq/tau, popplot.hmm.fft, color=pp['line_hmmcolor'],alpha=pp['line_ens_alpha'])
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
	hy = popplot.ax[0].hist(y[y>ymin],bins=pp['tc_nbins'],range=(rmin,rmax),histtype='stepfilled',density=True,color=pp['hist_color'])[0]
	# if pp['tc_showens']:
		# popplot.ax[0].axvline(x=np.log(popplot.ens.tc*tau),color='k')
	if pp['tc_showmean']:
		yy = y[y>ymin]
		yy = yy[yy>rmin]
		yy = yy[yy<rmax]
		qq = np.log(np.nanmean(np.exp(yy)))
		popplot.ax[0].axvline(x=qq,color='k',alpha=.9)

	if not gui.data.hmm_result is None and not popplot.hmm is None:
		hr = gui.data.hmm_result
		if hr.type in ['consensus vbfret']:
			popplot.ax[0].axvline(x=np.log(popplot.hmm.tc*tau),color=pp['line_hmmcolor'],alpha=.9)

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
	popplot.nmol = (y[y>ymin]).size

def plot_beta(gui,popplot,pp):
	beta = popplot.ind.beta.copy()
	tc = popplot.ind.tc.copy()
	# tf = popplot.ind.tfit.copy()
	# tf_cut = pp['tc_fitcut']
	# x = tf >= tf_cut
	x = np.isfinite(beta)
	beta = beta[x]
	popplot.ax[0].hist(beta,bins=pp['beta_nbins'],range=(0,2),histtype='stepfilled',color=pp['hist_color'])
	if pp['beta_showens']:
		popplot.ax[0].axvline(x=popplot.ens.beta,color='k')
	if pp['beta_showmean']:
		popplot.ax[0].axvline(x=np.nanmean(beta),color='k')
	popplot.ax[0].set_xlim(0,2.)
	popplot.nmol = beta.size

def plot_scatter(gui,popplot,pp):
	from ..supporting.autocorr import vgamma
	tau = pp['time_dt']
	beta = popplot.ind.beta.copy()
	tc = popplot.ind.tc.copy()
	# tf = popplot.ind.tfit.copy()
	tf_cut = pp['tc_fitcut']
	# x = (tf >= tf_cut)*np.isfinite(beta)
	x = np.isnan(beta)
	beta[x] = 1.
	popplot.ax[0].loglog(beta,tc*tau,'o',alpha=.5,color=pp['hist_color'])
	# x = (tf < tf_cut)
	# popplot.ax[0].loglog(beta[x],tc[x]*tau,'o',alpha=.5,color='r')
	# x = np.isnan(beta)
	# popplot.ax[0].loglog(np.ones(int(x.sum())),tc[x]*tau,'o',alpha=.5,color='r')

	popplot.ax[0].set_xlim(.1,3.)
	popplot.ax[0].set_ylim(tau,tau*popplot.fpb.shape[1])
	bb = np.linspace(.1,3.,10000)
	p = np.array((np.ones(bb.size),np.zeros(bb.size)+tf_cut,bb))
	tt = tf_cut/bb*vgamma(1./bb)*tau
	popplot.ax[0].plot(bb,tt,color='k',ls='--',lw=1.,alpha=.9)
	popplot.nmol = int(np.isfinite(beta).sum())

	if not gui.data.hmm_result is None and not popplot.hmm is None:
		hr = gui.data.hmm_result
		if hr.type in ['consensus vbfret']:
			popplot.ax[0].axhline(y=popplot.hmm.tc*tau,color=pp['line_hmmcolor'],alpha=.9)
			popplot.ax[0].axvline(x=popplot.hmm.beta,color=pp['line_hmmcolor'],alpha=.9)


def plot_indacf(gui,popplot,pp):
	from ..supporting.autocorr import fit_acf
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
	fit = fit_acf(popplot.ind.t,popplot.ind.y[ind],pp['tc_fit_ymin'],pp['fit_biexp'])
	popplot.ax[0].plot(popplot.ind.t*tau,fit(popplot.ind.t),color='r',alpha=pp['line_alpha'])

	popplot.ax[0].set_xscale('linear')
	if pp['time_scale'] == 'log':
		popplot.ax[0].set_xscale('log')
		if pp['time_min'] < pp['time_dt']:
			pp['time_min'] = pp['time_dt']
	popplot.ax[0].set_xlim(pp['time_min'],pp['time_max'])
	popplot.ax[0].set_ylim(pp['acorr_min'],pp['acorr_max'])

	fit.tau = float(tau)
	popplot.ax[0].set_title(r"$t_c=%.3f: $"%(fit.calc_tc()) + str(fit),fontsize=pp['label_fontsize'])
