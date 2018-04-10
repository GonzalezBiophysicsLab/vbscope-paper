from PyQt5.QtWidgets import QInputDialog,QFileDialog

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import wiener


class traj_container():
	def __init__(self,gui=None):
		self.gui = gui

		self.hmm_result = None
		self.d = None
		self.pre_list = np.array(())
		self.pb_list = np.array(())
		self.class_list = np.array((()))
		self.fret = np.array(())

		self.deadprob = None

	def cross_corr_order(self):

		x = self.d[:,0] #- self.d[:,0].mean(1)[:,None]
		y = self.d[:,1] #- self.d[:,1].mean(1)[:,None]
		x = np.gradient(x,axis=1)
		y = np.gradient(y,axis=1)

		a = np.fft.fft(x,axis=1)
		b = np.conjugate(np.fft.fft(y,axis=1))
		order = np.fft.ifft((a*b),axis=1)
		order = order[:,0].real.argsort()

		self.d = self.d[order]
		self.gui.log('Trajectories sorted by cross correlation',True)

	def update_fret(self):
		q = np.copy(self.d)
		if self.gui.prefs['wiener_smooth']:
			for i in range(q.shape[1]):
				for j in range(q.shape[0]):
					try:
						q[j,i] = wiener(q[j,i])
					except:
						pass
		bts = self.gui.prefs['bleedthrough'].reshape((4,4))
		for i in range(self.gui.ncolors):
			for j in range(self.gui.ncolors):
				q[:,j] -= bts[i,j]*q[:,i]
		# C-1,N,T
		self.fret = np.array([q[:,i]/q.sum(1) for i in range(1,self.gui.ncolors)])

	def safe_hmm(self):
		if not self.hmm_result is None:
			self.hmm_result = None
			try:
				self.gui.plot.plot_no_hmm()
				self.gui.plot.update_blits()
			except:
				pass
			self.safe_hmm()

	def cull_min(self,threshold=None):
		if threshold is None:
			threshold,success = QInputDialog.getDouble(self.gui,"Remove Traces with Min","Remove traces with values less than:",value=-10000)
		else:
			success = True
		if success:
			cut = np.min(self.d,axis=(1,2)) > threshold
			pbt = self.pb_list.copy()
			pret = self.pre_list.copy()

			d = self.d[cut]
			pbt = pbt[cut]
			pret = pret[cut]
			self.gui.plot.index = 0
			self.gui.initialize_data(d,sort=False)
			self.pb_list = pbt
			self.pre_list = pret
			self.gui.plot.initialize_plots()
			self.gui.initialize_sliders()

			msg = "Cull traces: kept %d out of %d = %f %%, with a value less than %f"%(cut.sum(),cut.size,cut.sum()/float(cut.size),threshold)
			self.gui.log(msg,True)
			self.gui.update_display_traces()

	def cull_max(self,event=None,threshold=None):
		if threshold is None:
			threshold,success = QInputDialog.getDouble(self.gui,"Remove Traces with Max","Remove traces with values greater than:",value=65535)
		else:
			success = True
		if success:
			cut = np.max(self.d,axis=(1,2)) < threshold
			pbt = self.pb_list.copy()
			pret = self.pre_list.copy()

			d = self.d[cut]
			pbt = pbt[cut]
			pret = pret[cut]
			self.gui.plot.index = 0
			self.gui.initialize_data(d,sort=False)
			self.pb_list = pbt
			self.pre_list = pret
			self.gui.plot.initialize_plots()
			self.gui.initialize_sliders()

			msg = "Cull traces: kept %d out of %d = %f %%, with a value greater than %f"%(cut.sum(),cut.size,cut.sum()/float(cut.size),threshold)
			self.gui.log(msg,True)
			self.gui.update_display_traces()

	## Remove trajectories with number of kept-frames < threshold
	def cull_pb(self):
		if not self.d is None and self.gui.ncolors == 2:
			self.safe_hmm()
			pbt = self.pb_list.copy()
			pret = self.pre_list.copy()
			dt = pbt-pret
			cut = dt > self.gui.prefs['min_length']

			d = self.d[cut]
			pbt = pbt[cut]
			pret = pret[cut]
			self.gui.plot.index = 0
			self.gui.initialize_data(d,sort=False)
			self.pb_list = pbt
			self.pre_list = pret
			self.gui.plot.initialize_plots()
			self.gui.initialize_sliders()

			msg = "Cull short traces: kept %d out of %d = %f"%(cut.sum(),cut.size,cut.sum()/float(cut.size))
			self.gui.log(msg,True)
			self.gui.update_display_traces()

	def cull_photons(self,color=None,threshold=None):
		if not self.d is None:
			combos = ['%d'%(i) for i in range(self.gui.ncolors)]
			combos.append('0+1')
			if color is None:
				c,success1 = QInputDialog.getItem(self.gui,"Color","Choose Color channel",combos,editable=False)
			else:
				c = color
				success1 = True
			if success1:
				self.safe_hmm()
				dd = self.get_fluor().sum(-1) ## N,C,D
				y = np.zeros(dd.shape[0])

				cc = c.split('+')
				for ccc in cc:
					y +=  dd[:,int(ccc)]

				if threshold is None:
					threshold,success2 = QInputDialog.getDouble(self.gui,"Photon Cutoff","Total number of photons required to keep a trajectory",value=1000.,min=0.,max=1e10,decimals=3)
				else:
					success2 = True
 				if success2:
					keep = y > threshold
					d = self.d[keep]
					self.gui.plot.index = 0
					self.gui.initialize_data(d)
					self.gui.plot.initialize_plots()
					self.gui.initialize_sliders()
					self.gui.log('Cull Photons: %d traces with less than %d total counts in channel %s removed'%((keep==0).sum(),threshold,c),True)
					self.gui.update_display_traces()


	def get_plot_data(self):
		# f = self.fret
		self.update_fret()
		fpb = self.fret.copy()
		for j in range(self.gui.ncolors-1):
			for i in range(fpb.shape[1]):
				fpb[j][i,:self.pre_list[i]] = np.nan
				fpb[j][i,self.pb_list[i]:] = np.nan
				# if self.gui.prefs['synchronize_start_flag'] != 'True':
				# 	fpb[j][i,:self.gui.prefs['plotter_min_time']] = np.nan
				# 	fpb[j][i,self.gui.prefs['plotter_max_time']:] = np.nan

		checked = self.gui.classes_get_checked()
		fpb = fpb[:,checked]

		return fpb

	def remove_acceptor_bleach_from_fret(self):
		from ..supporting.photobleaching import get_point_pbtime

		self.update_fret()
		# fpb = self.fret.copy()
		d = self.get_fluor()
		checked = self.gui.classes_get_checked()
		if self.gui.ncolors == 2:
			# for i in range(fpb.shape[1]):
			for i in range(d.shape[0]):
				if checked[i]:
					# ff = fpb[0][i,self.pre_list[i]:self.pb_list[i]].copy()
					ff = d[i,1,self.pre_list[i]:self.pb_list[i]]
					pbt = get_point_pbtime(ff,1.,1.,1.,1000.)
					self.pb_list[i] = self.pre_list[i]+pbt
		self.gui.log('Removed acceptor bleaching from FRET range',True)



	def get_viterbi_data(self,signal=False):
		if not self.hmm_result is None:
			v = np.empty_like(self.fret[0]) + np.nan
			for i in range(v.shape[0]):
				if self.hmm_result.ran.count(i) > 0:
					ii = self.hmm_result.ran.index(i)
					if self.hmm_result.type == 'consensus vbfret':
						vi = self.hmm_result.result.viterbi[ii]
						if signal:
							vi = self.hmm_result.result.m[vi]
						v[i,self.pre_list[i]:self.pb_list[i]] = vi
					elif self.hmm_result.type == 'vb' or self.hmm_result.type == 'ml':
						r = self.hmm_result.results[ii]
						vi = r.viterbi
						if signal:
							vi = r.mu[vi]
						v[i,self.pre_list[i]:self.pb_list[i]] = vi

			# if self.gui.prefs['synchronize_start_flag'] != 'True':
			# 	v[i,:self.gui.prefs['plotter_min_time']] = np.nan
			# 	v[i,self.gui.prefs['plotter_max_time']:] = np.nan

			checked = self.gui.classes_get_checked()
			v = v[checked]
			return v
		else:
			return None

	def get_nstates(self,nstates=None):
		if nstates is None:
			nstates,success = QInputDialog.getInt(self.gui,"Number of States","Number of States",min=1)
		else:
			success = True
		return success,nstates

	def hmm_get_colorchannel(self,color=None):
		combos = ['%d'%(i) for i in range(self.gui.ncolors)]
		combos.append('Sum Intensity')
		combos.insert(0,'E_FRET')
		if color is None:
			color,success = QInputDialog.getItem(self.gui,"Color","Choose Color channel",combos,editable=False,current=0)
		else:
			success = True
		return success,color

	def hmm_get_traces(self,color='0'):
		if color.isdigit():
			z = self.get_fluor()[:,int(color)]
		elif color == 'Sum Intensity':
			z = self.get_fluor().sum(1)
		elif color == 'E_FRET':
			self.update_fret()
			z = self.fret[0]
		else:
			raise Exception('wtf color do you want?')

		y = []
		ran = []
		checked = self.gui.classes_get_checked()

		for i in range(z.shape[0]):
			if checked[i]:
				yy = z[i,self.pre_list[i]:self.pb_list[i]].copy()
				if color == 'E_FRET': ## Clip traces and redistribute randomly
					bad = np.bitwise_or((yy < -1.),np.bitwise_or((yy > 2),np.isnan(yy)))
					yy[bad] = np.random.uniform(low=-1,high=2,size=int(bad.sum()))
				if yy.size > 5:
					y.append(yy)
					ran.append(i)
		return y,ran

	def hmm_export(self):
		import cPickle as pickle
		oname = QFileDialog.getSaveFileName(self.gui, 'Export HMM results', '_HMM.dat','*.dat')
		if oname[0] != "":
			try:
				f = open(oname[0],'w')
				pickle.dump(self.hmm_result, f)
				f.close()
				self.gui.log('Exported HMM results as %s'%(oname[0]),True)
			except:
				QMessageBox.critical(self,'Export Traces','There was a problem trying to export the HMM results')
				self.gui.log('Failed to export HMM results as %s'%(oname[0]),True)

	def hmm_savechopped(self):
			oname = QFileDialog.getSaveFileName(self.gui, 'Save Chopped Traces', '_chopped.dat','*.dat')
			if oname[0] == "":
				return

			## N,C,T
			out = None

			from scipy.ndimage import label as ndilabel
			from scipy.ndimage import find_objects as ndifind

			## If it has an HMM
			for j in range(len(self.hmm_result.ran)): ## hmm index
				i = self.hmm_result.ran[j] ## trace index

				v = self.hmm_result.viterbi[j]

				pre = int(self.pre_list[i])
				post = int(self.pb_list[i])
				q = self.d[i,:,pre:post]

				labels,numlabels = ndilabel(v)
				slices = ndifind(labels)
				if len(slices)>0:
					for ss in slices:
						ss = ss[0]
						tmp = np.zeros((1,self.d.shape[1],self.d.shape[2]))
						tmp[0,:,:ss.stop-ss.start] = self.d[i,:,pre+ss.start:pre+ss.stop]
						tmp_cl = np.array((0,0,ss.stop-ss.start,0,ss.stop-ss.start))[None,:]
						if out is None:
							out = tmp.copy()
							classes = tmp_cl.copy()
						else:
							out = np.append(out,tmp,axis=0)
							classes = np.append(classes,tmp_cl,axis=0)

			q = np.zeros((out.shape[0]*out.shape[1],out.shape[2]))
			for i in range(out.shape[1]):
				q[i::out.shape[1]] = out[:,i]
			np.savetxt(oname[0],q.T,delimiter=',')
			np.savetxt(oname[0][:-4]+"_classes.dat",classes.astype('i'),delimiter=',')
			self.gui.log('Exported chopped traces as %s'%(oname[0]),True)

	def _cancel_run(self):
		self.flag_running = False

	def run_conhmm(self,nstates=None,color=None):
		self.gui.set_status('Compiling...')
		from ..supporting.hmms.consensus_vb_em_hmm import consensus_vb_em_hmm,consensus_vb_em_hmm_parallel
		self.gui.set_status('')

		if not self.d is None:
			success1,nstates = self.get_nstates(nstates)
			if not success1:
				return
			success2,color = self.hmm_get_colorchannel(color)

			if success1 and success2:
				try:
					y,ran = self.hmm_get_traces(color)
				except:
					return


				priors = np.array([self.gui.prefs[sss] for sss in ['vb_prior_beta','vb_prior_a','vb_prior_b','vb_prior_pi','vb_prior_alpha']])
				self.hmm_result = consensus_hmm_result()
				self.hmm_result.type = 'consensus vbfret'
				self.gui.set_status('Running...')
				self.hmm_result.result = consensus_vb_em_hmm_parallel(y,nstates,maxiters=self.gui.prefs['hmm_max_iters'],threshold=self.gui.prefs['hmm_threshold'],nrestarts=self.gui.prefs['hmm_nrestarts'],prior_strengths=priors,ncpu=self.gui.prefs['ncpu'])

				self.gui.log(self.hmm_result.result.report(),True)
				# self.gui.log(result.gen_report(tau=self.gui.prefs['tau']))
				self.hmm_result.ran = ran
				self.gui.plot.initialize_hmm_plot()
				self.gui.plot.update_plots()

				self.hmm_export()
				if self.gui.prefs['hmm_binding_expt'] is True:
					self.hmm_savechopped()

	def run_vbhmm(self,nstates=None,color=None):
		self.gui.set_status('Compiling...')
		from ..supporting.hmms.vb_em_hmm import vb_em_hmm,vb_em_hmm_parallel
		# from ..supporting.hmms.ml_em_gmm import ml_em_gmm
		self.gui.set_status('')

		if not self.d is None:
			success1,nstates = self.get_nstates(nstates)
			if not success1:
				return
			success2,color = self.hmm_get_colorchannel(color)

			if success1 and success2:
				try:
					y,ran = self.hmm_get_traces(color)
				except:
					return

				from ..ui.ui_progressbar import progressbar
				prog = progressbar()
				prog.setRange(0,len(y))
				prog.setWindowTitle('vbFRET HMM Progress')
				prog.setLabelText('Current Trajectory')
				self.flag_running = True
				prog.canceled.connect(self._cancel_run)
				prog.show()

				priors = np.array([self.gui.prefs[sss] for sss in ['vb_prior_beta','vb_prior_a','vb_prior_b','vb_prior_pi','vb_prior_alpha']])
				self.hmm_result = ensemble_hmm_result()
				self.hmm_result.type = 'vb'
				for i in range(len(y)):
					prog.setValue(i)
					self.gui.app.processEvents()
					if self.flag_running:
						self.hmm_result.results.append(vb_em_hmm_parallel(y[i],nstates,maxiters=self.gui.prefs['hmm_max_iters'],threshold=self.gui.prefs['hmm_threshold'],nrestarts=self.gui.prefs['hmm_nrestarts'],prior_strengths=priors,ncpu=self.gui.prefs['ncpu']))
				self.hmm_result.ran = ran[:len(self.hmm_result.results)]

				self.gui.log("HMM report - %d states"%(nstates),True)
				self.gui.log("Finished %d trajectories"%(len(self.hmm_result.results)),True)
				# self.gui.log(result.gen_report(tau=self.gui.prefs['tau']))

				self.gui.plot.initialize_hmm_plot()
				self.gui.plot.update_plots()

				self.hmm_export()
				if self.gui.prefs['hmm_binding_expt'] is True:
					self.hmm_savechopped()

	def run_mlhmm(self,nstates=None,color=None):
		self.gui.set_status('Compiling...')
		from ..supporting.hmms.ml_em_hmm import ml_em_hmm, ml_em_hmm_parallel
		# from ..supporting.hmms.ml_em_gmm import ml_em_gmm
		self.gui.set_status('')

		if not self.d is None:
			success1,nstates = self.get_nstates(nstates)
			if not success1:
				return
			success2,color = self.hmm_get_colorchannel(color)

			if success1 and success2:
				try:
					y,ran = self.hmm_get_traces(color)
				except:
					return

				from ..ui.ui_progressbar import progressbar
				prog = progressbar()
				prog.setRange(0,len(y))
				prog.setWindowTitle('ML HMM Progress')
				prog.setLabelText('Current Trajectory')
				self.flag_running = True
				prog.canceled.connect(self._cancel_run)
				prog.show()

				self.hmm_result = ensemble_hmm_result()
				self.hmm_result.type = 'ml'
				for i in range(len(y)):
					if self.flag_running:
						self.hmm_result.results.append(ml_em_hmm_parallel(y[i],nstates,maxiters=self.gui.prefs['hmm_max_iters'],threshold=self.gui.prefs['hmm_threshold'],nrestarts=self.gui.prefs['hmm_nrestarts'],ncpu=self.gui.prefs['ncpu']))
					prog.setValue(i)
					self.gui.app.processEvents()
				self.hmm_result.ran = ran[:len(self.hmm_result.results)]

				self.gui.log("HMM report - %d states"%(nstates),True)
				self.gui.log("Finished %d trajectories"%(len(self.hmm_result.results)),True)
				# self.gui.log(result.gen_report(tau=self.gui.prefs['tau']))

				self.gui.plot.initialize_hmm_plot()
				self.gui.plot.update_plots()

				self.hmm_export()
				if self.gui.prefs['hmm_binding_expt'] is True:
					self.hmm_savechopped()

	def remove_beginning(self,nd=None):
		if not self.d is None:
			self.safe_hmm()
			if nd is None:
				nd,success = QInputDialog.getInt(self.gui,"Remove Datapoints","Number of datapoints to remove starting from the beginning of the movie")
			else:
				success = True
			if success and nd > 1 and nd < self.d.shape[2]:

				self.gui.plot.index = 0
				self.gui.initialize_data(self.d[:,:,nd:])
				self.gui.log('Removed %d datapoints from start of all traces'%(nd),True)
				self.gui.plot.initialize_plots()
				self.gui.initialize_sliders()

	def photobleach_step(self):
		if not self.d is None and self.gui.ncolors == 2:
			self.safe_hmm()
			from ..supporting.photobleaching import pb_ensemble
			q = np.copy(self.d)
			q[:,1] -= self.gui.prefs['bleedthrough'][1]*q[:,0]
			# l1 = calc_pb_time(self.fret,self.gui.prefs['min_length_cutoff'])
			if self.gui.prefs['photobleaching_flag'] is True:
				qq = q[:,0] + q[:,1]
			else:
				qq = q[:,1]
			l2 = pb_ensemble(qq)[1]
			# self.pb_list = np.array([np.min((l1[i],l2[i])) for i in range(l1.size)])
			self.pb_list = l2
			self.gui.log('Automated photobleaching ensemble method ran',True)
			self.gui.plot.update_plots()

	def get_fluor(self):
		q = np.copy(self.d)
		bts = self.gui.prefs['bleedthrough'].reshape((4,4))
		for i in range(self.gui.ncolors):
			for j in range(self.gui.ncolors):
				q[:,j] -= bts[i,j]*q[:,i]
		return q

	def estimate_mvs(self):
		m0s = np.array(())
		v0s = np.array(())
		m1s = np.array(())
		v1s = np.array(())
		fracs = np.array(())

		ss = 5.

		checked = self.gui.classes_get_checked()
		data = self.d[checked]

		for i in range(data.shape[1]):
			dd = data[:,i]
			## estimate from min
			from ..supporting.normal_minmax_dist import estimate_from_min, backout_var_fixed_m
			m,v = estimate_from_min(dd.min(1),dd.shape[1])
			## cleanup estimate
			dr = dd[dd > (m-ss*np.sqrt(v))]
			dr = dr[dr < (m+ss*np.sqrt(v))]
			if dr.size > 0:
				m = np.median(dr)
				# v = backout_var_fixed_m(dd.min(1),dd.shape[1])
				v = np.square(m - np.percentile(dr,50.0 - 68.27/2))

			m1 = np.median(dd[dd > m + ss*np.sqrt(v)])
			v1 = np.var(dd[dd > m + ss*np.sqrt(v)])
			m0s = np.append(m0s,m)
			v0s = np.append(v0s,v)
			m1s = np.append(m1s,m1)
			v1s = np.append(v1s,v1)

			p = normal(dd[:,None].flatten()[:,None],np.array((m0s[i],m1s[i]))[None,:],np.array((v0s[i],v1s[i]))[None,:])
			p = p / p.sum(1)[:,None]
			frac = p.sum(0)/p.sum()
			fracs = np.append(fracs,frac)

		dd = data.sum(1)
		m1 = np.mean(dd[dd > m0s.sum() + ss*np.sqrt(v0s.sum())])
		v1 = np.var(dd[dd > m0s.sum() + ss*np.sqrt(v0s.sum())])
		p = normal(dd[:,None].flatten()[:,None],np.array((m0s.sum(),m1))[None,:],np.array((v0s.sum(),v1))[None,:])
		p = p / p.sum(1)[:,None]
		frac = p.sum(0)/p.sum()

		m0s = np.append(m0s,m0s.sum())
		v0s = np.append(v0s,v0s.sum())
		m1s = np.append(m1s,m1)
		v1s = np.append(v1s,v1)
		fracs = np.append(fracs,frac)
		return m0s,v0s,m1s,v1s,fracs

	def posterior_sum(self):
		m0s,v0s,m1s,v1s,fracs = self.estimate_mvs()
		ms = np.array((m0s[-1],m1s[-1]))
		vs = np.array((v0s[-1],v1s[-1]))
		fracs = fracs[-2:]

		checked = self.gui.classes_get_checked()
		## K,N,(C),T
		dd = np.sum(self.d[checked],axis=1)
		p = fracs[:,None,None]*normal(dd[None,:,:],ms[:,None,None],vs[:,None,None])
		p /= p.sum(0)[None,:,:]

		## number (soft) of datapoints that aren't bg class
		self.deadprob = p.sum(-1)[1]

	def remove_dead(self,threshold = None):
		if threshold is None:
			threshold,success2 = QInputDialog.getDouble(self.gui,"Soft Frame Cutoff","Soft number of frames with signal required to keep a trajectory",value=20.,min=0.,max=self.d.shape[2],decimals=3)
		else:
			success2 = True
		if success2:
			self.posterior_sum()
			dp  = self.deadprob.copy()
			keep = self.deadprob > threshold
			self.gui.plot.index = 0
			self.gui.initialize_data(self.d[keep])
			self.gui.data.deadprob = dp
			self.gui.plot.initialize_plots()
			self.gui.initialize_sliders()
			self.gui.log('Remove Dead Frames: %d traces with less than %d total frames in the summed channel removed'%((keep==0).sum(),threshold),True)
			self.gui.update_display_traces()

def normal(x,m,v):
	return 1./np.sqrt(2.*np.pi*v) * np.exp(-.5/v*(x-m)**2.)


class ensemble_hmm_result(object):
	def __init__(self, *args):
		self.args = args
		self.results = []

class consensus_hmm_result(object):
	def __init__(self, *args):
		self.args = args
		self.result = None
