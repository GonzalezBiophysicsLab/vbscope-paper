from PyQt5.QtWidgets import QInputDialog,QFileDialog,QMessageBox
import os
import matplotlib.pyplot as plt
import numpy as np



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

	def filter(self,y):
		from scipy import signal
		from scipy.ndimage import gaussian_filter1d,median_filter

		method = self.gui.prefs['filter_method']
		w = float(self.gui.prefs['filter_width'])

		if w <= 0: w = .01
		if method == 0:
			return gaussian_filter1d(y,w)
		elif method == 1:
			w = int(np.floor(w/2)*2+1)
			w = np.max((w,3))
			return signal.wiener(y,mysize=w)
		elif method == 2:
			w = int(np.max((1,w)))
			return median_filter(y,w)
		else:
			if w < 2: w = 2.
			b, a = signal.bessel(8, 1./w)
			return signal.filtfilt(b, a, y)

	def calc_cross_corr(self,d=None): ## of gradient
		try:
			if d is None:
				x = self.d[:,0] #- self.d[:,0].mean(1)[:,None]
				y = self.d[:,1] #- self.d[:,1].mean(1)[:,None]
			else:
				x = d[0].reshape((1,-1))
				y = d[1].reshape((1,-1))
			x = np.gradient(x,axis=1)
			y = np.gradient(y,axis=1)

			a = np.fft.fft(x,axis=1)
			b = np.conjugate(np.fft.fft(y,axis=1))
			cc = np.fft.ifft((a*b),axis=1)
			cc = cc[:,0].real
			return cc
			# return np.abs(cc)
			#return cc.argsort()
		except:
			return np.zeros_like(x)

	def calc_all_cc(self):
		self.cc_list = np.array([self.calc_cross_corr(self.d[j,:,self.pre_list[j]:self.pb_list[j]])[0] for j in range(self.d.shape[0])])

	def cross_corr_order(self):
		self.calc_all_cc()
		order = self.cc_list.argsort()
		self.d = self.d[order]
		self.pre_list = self.pre_list[order]
		self.pb_list = self.pb_list[order]
		self.class_list = self.class_list[order]
		if not self.hmm_result is None:
			if self.hmm_result.type is 'consensus vbfret':
				self.hmm_result.result.r = self.hmm_result.result.r[order]
				self.hmm_result.result.viterbi = self.hmm_result.result.viterbi[order]
				self.hmm_result.ran = self.hmm_result.ran[order]
			else:
				self.hmm_result.results = self.hmm_result.results[order]
				self.hmm_result.ran = self.hmm_result.ran[order]

		self.safe_hmm()
		self.update_fret()
		self.gui.plot.update_plots()
		self.gui.update_display_traces()
		self.gui.log('Trajectories sorted by cross correlation',True)

	def update_fret(self,filter_flag=False):
		q = np.copy(self.d)
		if filter_flag:
			for i in range(q.shape[1]):
				for j in range(q.shape[0]):
					try:
						q[j,i] = self.filter(q[j,i])
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
			keep = np.min(self.d,axis=(1,2)) > threshold
			if self.remove_traces(keep):
				self.gui.plot.initialize_plots()
				self.gui.initialize_sliders()

				msg = "Cull traces: kept %d out of %d = %f %%, with a value less than %f"%(keep.sum(),keep.size,keep.sum()/float(keep.size),threshold)
				self.gui.log(msg,True)
				self.gui.update_display_traces()

	def cull_max(self,event=None,threshold=None):
		if threshold is None:
			threshold,success = QInputDialog.getDouble(self.gui,"Remove Traces with Max","Remove traces with values greater than:",value=65535)
		else:
			success = True
		if success:
			keep = np.max(self.d,axis=(1,2)) < threshold
			if self.remove_traces(keep):
				self.gui.plot.initialize_plots()
				self.gui.initialize_sliders()

				msg = "Cull traces: kept %d out of %d = %f %%, with a value greater than %f"%(keep.sum(),keep.size,keep.sum()/float(keep.size),threshold)
				self.gui.log(msg,True)
				self.gui.update_display_traces()

	## Remove trajectories with number of kept-frames < threshold
	def cull_pb(self):
		if not self.d is None and self.gui.ncolors == 2:
			self.safe_hmm()
			dt = self.pb_list-self.pre_list
			keep = dt > self.gui.prefs['min_length']

			if self.remove_traces(keep):
				self.gui.plot.initialize_plots()
				self.gui.initialize_sliders()

				msg = "Cull short traces: kept %d out of %d = %f"%(keep.sum(),keep.size,keep.sum()/float(keep.size))
				self.gui.log(msg,True)
				self.gui.update_display_traces()

	def remove_traces(self,mask):
		if mask.sum() == 0:
			msg = "ERROR: cannot remove all traces"
			self.gui.log(msg,True)
			return False
		else:
			d = self.d[mask]
			pbt = self.pb_list[mask].copy()
			pret = self.pre_list[mask].copy()
			classes = self.class_list[mask].copy()
			self.gui.plot.index = 0
			self.gui.initialize_data(d,sort=False)
			self.pb_list = pbt
			self.pre_list = pret
			self.class_list = classes
			return True

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
					if self.remove_traces(keep):
						self.gui.plot.initialize_plots()
						self.gui.initialize_sliders()
						self.gui.log('Cull Photons: %d traces with less than %d total counts in channel %s removed'%((keep==0).sum(),threshold,c),True)
						self.gui.update_display_traces()


	def get_plot_data(self,filter_flag=False):
		# f = self.fret
		self.update_fret(filter_flag)
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

		d = self.get_fluor()
		checked = self.gui.classes_get_checked()
		if self.gui.ncolors == 2:
			for i in range(d.shape[0]):
				if checked[i]:
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

	def get_nstate_range(self,nmin=None,nmax=None):
		if nmin is None or nmax is None:
			nmin,success1 = QInputDialog.getInt(self.gui,"Number of States","Minimum Number of States",min=1,value=1)
			nmax,success2 = QInputDialog.getInt(self.gui,"Number of States","Maximum Number of States",min=1,value=6)
		else:
			success1 = True
			success2 = True
		return success1,success2,nmin,nmax

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
			self.update_fret(filter_flag=self.gui.prefs['hmm_filter'])
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

	def hmm_export(self,prompt_export=True,oname=None):
		if prompt_export:
			oname = QFileDialog.getSaveFileName(self.gui, 'Export HMM results', '_HMM.hdf5','*.hdf5')
		else:
			oname = [oname]

		if oname[0] != "" and not oname[0] is None:
			try:
				def _addhash(hdf5_item):
					"""
					Acts on an h5py item to add identification attributes:
						* Time Created
						* Hash ID
					Input:
						* `hdf5_item` is an h5py item (e.g., file, group, or dataset)
					"""
					from time import ctime
					from hashlib import md5

					time = ctime()
					hdf5_item.attrs['Time Created'] = time
					# h5py items don't really hash, so.... do this, instead.
					# Should be unique, and the point of the hash is for identification
					hdf5_item.attrs['Unique ID'] = md5((time + str(hdf5_item.id.id) + str(np.random.rand())).encode('utf-8')).hexdigest()

				import h5py
				hr = self.hmm_result
				f = open(oname[0],'w')
				f.close()
				f = h5py.File(oname[0],'w')
				_addhash(f)
				f.attrs['type'] = str(hr.type)
				f.attrs['log'] = self.gui._log.textedit.toPlainText()
				f.flush()
				ran = np.array(hr.ran,dtype='int')
				f.create_dataset('ran',ran.shape,ran.dtype,ran)
				_addhash(f['ran'])
				f.flush()

				f.create_group("models")
				if hr.type == 'consensus vbfret':
					for i in range(len(hr.models)):
						g = f.create_group("models/model_%08d"%(i))
						_addhash(g)
						m = hr.models[i]
						for key,value in list(m.__dict__.items()):
							if type(value) is np.ndarray:
								g.create_dataset(key,data=value)
						g.create_dataset('iteration',data=np.array(m.iteration))
						gg = g.create_group('r')
						for j in range(len(m.r)):
							gg.create_dataset('trace_%08d'%(j),data=m.r[j])
						gg = g.create_group('viterbi')
						for j in range(len(m.viterbi)):
							gg.create_dataset('trace_%08d'%(j),data=m.viterbi[j])
				else:
					for j in range(len(hr.models)):
						for i in range(len(hr.models[j])):
							g = f.create_group("models/trace_%08d/model_%08d"%(j,i))
							_addhash(g)
							m = hr.models[j][i]
							for key,value in list(m.__dict__.items()):
								if type(value) is np.ndarray:
									g.create_dataset(key,data=value)
							g.create_dataset('iteration',data=np.array(m.iteration))
				f.flush()
				f.close()

				# import cPickle as pickle
				# f = open(oname[0],'w')
				# pickle.dump(self.hmm_result, f)
				# f.close()
				self.gui.log('Exported HMM results as %s'%(oname[0]),True)
		except:
				QMessageBox.critical(self.gui,'Export Traces','There was a problem trying to export the HMM results')
				self.gui.log('Failed to export HMM results as %s'%(oname[0]),True)

	def hmm_load(self,fname):
		import h5py
		from ..supporting.hmms.fxns.hmm_related import result_consensus_bayesian_hmm, result_bayesian_hmm, result_ml_hmm

		f = h5py.File(fname,'r')
		t = f.attrs['type']
		if t.startswith('consensus'):
			self.hmm_result = consensus_hmm_result()
		else:
			self.hmm_result = ensemble_hmm_result()

		self.hmm_result.type = t
		self.hmm_result.ran = f['ran'].value.tolist()
		self.hmm_result.models = []

		bayes_list = ['r','a','b','m','beta','pi','tmatrix','E_lnlam','E_lnpi','E_lntm','likelihood','iteration']
		ml_list = ['mu','var','r','ppi','tmatrix','likelihood','iteration']

		if self.hmm_result.type.startswith('consensus'):
			for g in list(f['models'].values()):
				r = [rr.value for rr in list(g['r'].values())]
				m = result_consensus_bayesian_hmm(r,*[g[v].value for v in bayes_list[1:]])
				m.viterbi = [v.value for v in list(g['viterbi'].values())]
				self.hmm_result.models.append(m)
			self.likelihoods = np.array([m.likelihood[-1,0] for m in self.hmm_result.models])
			self.hmm_result.result = self.hmm_result.models[np.argmax(self.likelihoods)]

		elif self.hmm_result.type == 'vb':
			self.hmm_result.results = []
			for g in list(f['models'].values()):
				trace = []
				for gg in list(g.values()):
					m = result_bayesian_hmm(*[gg[v].value for v in bayes_list])
					m.viterbi = gg['viterbi'].value
					trace.append(m)
				likelihoods = np.array([m.likelihood[-1,0] for m in trace])
				self.hmm_result.results.append(trace[np.argmax(likelihoods)])
				self.hmm_result.models.append(trace)

		elif self.hmm_result.type == 'ml':
			self.hmm_result.results = []
			for g in list(f['models'].values()):
				trace = []
				for gg in list(g.values()): ## There's only one, but be general here...
					m = result_ml_hmm(*[gg[v].value for v in ml_list])
					m.viterbi = gg['viterbi'].value
					trace.append(m)
				self.hmm_result.models.append(trace)
				self.hmm_result.results.append(trace[0])

		f.close()

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

				if self.hmm_result.type == 'consensus vbfret':
					v = self.hmm_result.viterbi[j]
				else:
					v = self.hmm_result.results[j].viterbi

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

	def run_conhmm_model(self,nmin=None,nmax=None,color=None,prompt_export=True):
		self.gui.set_status('Compiling...')
		from ..supporting.hmms.consensus_vb_em_hmm import consensus_vb_em_hmm,consensus_vb_em_hmm_parallel
		# from ..supporting.hmms.ml_em_gmm import ml_em_gmm
		self.gui.set_status('')

		if not self.d is None:
			success1,success2,nmin,nmax = self.get_nstate_range(nmin,nmax)
			if not success1 or not success2:
				return
			success3,color = self.hmm_get_colorchannel(color)

			if success1 and success2 and success3:
				try:
					y,ran = self.hmm_get_traces(color)
				except:
					return

				from ..ui.ui_progressbar import progressbar
				prog = progressbar()
				prog.setRange(nmin,nmax+1)
				prog.setWindowTitle('Consensus HMM Progress')
				self.flag_running = True
				prog.canceled.connect(self._cancel_run)
				prog.show()

				priors = np.array([self.gui.prefs[sss] for sss in ['vb_prior_beta','vb_prior_a','vb_prior_b','vb_prior_pi','vb_prior_alpha']])
				self.hmm_result = consensus_hmm_result()
				self.hmm_result.type = 'consensus vbfret'
				self.hmm_result.models = []
				self.hmm_result.likelihoods = []
				for i in range(nmin,nmax+1):
					prog.setValue(i)
					prog.setLabelText('Current Model: %d'%(i))
					self.gui.app.processEvents()

					if self.flag_running:
						rs = consensus_vb_em_hmm_parallel(y,i,maxiters=self.gui.prefs['hmm_max_iters'],threshold=self.gui.prefs['hmm_threshold'],nrestarts=self.gui.prefs['hmm_nrestarts'],prior_strengths=priors,ncpu=self.gui.prefs['ncpu'])
						ls = rs.likelihood[-1,0]

						self.hmm_result.models.append(rs)
						self.hmm_result.likelihoods.append(ls)
				if i == nmax:
					modelmax = np.argmax(self.hmm_result.likelihoods)
					self.hmm_result.result = self.hmm_result.models[modelmax]

					self.gui.log("HMM report - %d to %d model selection"%(nmin,nmax),True)
					self.gui.log("Best Model - %d states"%(self.hmm_result.result.mu.size))
					self.gui.log(self.hmm_result.result.report(),True)

					self.hmm_result.ran = ran

					self.gui.plot.initialize_hmm_plot()
					self.gui.plot.update_plots()

					self.hmm_export(prompt_export)
					if self.gui.prefs['hmm_binding_expt'] is True:
						self.hmm_savechopped()
				else:
					self.safe_hmm()


	def run_conhmm(self,nstates=None,color=None,prompt_export=True):
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
				self.hmm_result.models = []
				self.gui.set_status('Running...')
				self.hmm_result.result = consensus_vb_em_hmm_parallel(y,nstates,maxiters=self.gui.prefs['hmm_max_iters'],threshold=self.gui.prefs['hmm_threshold'],nrestarts=self.gui.prefs['hmm_nrestarts'],prior_strengths=priors,ncpu=self.gui.prefs['ncpu'])
				self.hmm_result.models = [self.hmm_result.result]

				self.gui.log(self.hmm_result.result.report(),True)
				# self.gui.log(result.gen_report(tau=self.gui.prefs['tau']))
				self.hmm_result.ran = ran
				self.gui.plot.initialize_hmm_plot()
				self.gui.plot.update_plots()

				self.hmm_export(prompt_export)
				if self.gui.prefs['hmm_binding_expt'] is True:
					self.hmm_savechopped()

	def run_vbhmm_model(self,nmin=None,nmax=None,color=None,prompt_export=True):
		self.gui.set_status('Compiling...')
		from ..supporting.hmms.vb_em_hmm import vb_em_hmm,vb_em_hmm_parallel,vb_em_hmm_model_selection_parallel
		# from ..supporting.hmms.ml_em_gmm import ml_em_gmm
		self.gui.set_status('')

		if not self.d is None:
			success1,success2,nmin,nmax = self.get_nstate_range(nmin,nmax)
			if not success1 or not success2:
				return
			success3,color = self.hmm_get_colorchannel(color)

			if success1 and success2 and success3:
				try:
					y,ran = self.hmm_get_traces(color)
				except:
					return

				from ..ui.ui_progressbar import progressbar
				prog = progressbar()
				prog.setRange(0,len(y))
				prog.setWindowTitle('vbFRET HMM Progress')
				self.flag_running = True
				prog.canceled.connect(self._cancel_run)
				prog.show()

				priors = np.array([self.gui.prefs[sss] for sss in ['vb_prior_beta','vb_prior_a','vb_prior_b','vb_prior_pi','vb_prior_alpha']])
				self.hmm_result = ensemble_hmm_result()
				self.hmm_result.type = 'vb'
				self.hmm_result.models = []
				self.hmm_result.likelihoods = []
				for i in range(len(y)):
					prog.setValue(i)
					prog.setLabelText('Current Trajectory: %d/%d'%(i,len(y)))
					self.gui.app.processEvents()

					if self.flag_running:
						rs,ls = vb_em_hmm_model_selection_parallel(y[i],nmin=nmin,nmax=nmax,maxiters=self.gui.prefs['hmm_max_iters'],threshold=self.gui.prefs['hmm_threshold'],nrestarts=self.gui.prefs['hmm_nrestarts'],prior_strengths=priors,ncpu=self.gui.prefs['ncpu'])
						self.hmm_result.models.append(rs)
						self.hmm_result.likelihoods.append(ls)
						modelmax = np.argmax(ls)
						self.hmm_result.results.append(rs[modelmax])
				self.hmm_result.ran = ran[:len(self.hmm_result.results)]

				self.gui.log("HMM report - %d to %d model selection"%(nmin,nmax),True)
				self.gui.log("Finished %d trajectories"%(len(self.hmm_result.results)),True)
				# self.gui.log(result.gen_report(tau=self.gui.prefs['tau']))

				self.gui.plot.initialize_hmm_plot()
				self.gui.plot.update_plots()

				self.hmm_export(prompt_export)
				if self.gui.prefs['hmm_binding_expt'] is True:
					self.hmm_savechopped()

	def run_vbhmm(self,nstates=None,color=None,prompt_export=True):
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
				self.hmm_result.models= []
				for i in range(len(y)):
					prog.setValue(i)
					prog.setLabelText('Current Trajectory: %d/%d'%(i,len(y)))
					self.gui.app.processEvents()
					if self.flag_running:
						self.hmm_result.results.append(vb_em_hmm_parallel(y[i],nstates,maxiters=self.gui.prefs['hmm_max_iters'],threshold=self.gui.prefs['hmm_threshold'],nrestarts=self.gui.prefs['hmm_nrestarts'],prior_strengths=priors,ncpu=self.gui.prefs['ncpu']))
						self.hmm_result.models.append([self.hmm_result.results[-1]])
				self.hmm_result.ran = ran[:len(self.hmm_result.results)]

				self.gui.log("HMM report - %d states"%(nstates),True)
				self.gui.log("Finished %d trajectories"%(len(self.hmm_result.results)),True)
				# self.gui.log(result.gen_report(tau=self.gui.prefs['tau']))

				self.gui.plot.initialize_hmm_plot()
				self.gui.plot.update_plots()

				self.hmm_export(prompt_export)
				if self.gui.prefs['hmm_binding_expt'] is True:
					self.hmm_savechopped()

	def run_mlhmm(self,nstates=None,color=None,prompt_export=True):
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
				self.hmm_result.models = []
				for i in range(len(y)):
					if self.flag_running:
						self.hmm_result.results.append(ml_em_hmm_parallel(y[i],nstates,maxiters=self.gui.prefs['hmm_max_iters'],threshold=self.gui.prefs['hmm_threshold'],nrestarts=self.gui.prefs['hmm_nrestarts'],ncpu=self.gui.prefs['ncpu']))
						self.hmm_result.models.append([self.hmm_result.results[-1]])

					prog.setValue(i)
					prog.setLabelText('Current Trajectory: %d/%d'%(i,len(y)))
					self.gui.app.processEvents()
				self.hmm_result.ran = ran[:len(self.hmm_result.results)]

				self.gui.log("HMM report - %d states"%(nstates),True)
				self.gui.log("Finished %d trajectories"%(len(self.hmm_result.results)),True)
				# self.gui.log(result.gen_report(tau=self.gui.prefs['tau']))

				self.gui.plot.initialize_hmm_plot()
				self.gui.plot.update_plots()

				self.hmm_export(prompt_export)
				if self.gui.prefs['hmm_binding_expt'] is True:
					self.hmm_savechopped()

	def run_biasd(self,color=None):
		import sys
		pp = self.gui.prefs
		if not pp['biasd_path'] in sys.path:
			sys.path.append(pp['biasd_path'])
		try:
			import biasd as b
		except:
			self.gui.set_status("Failed to load BIASD; Check biasd_path preference")
			QMessageBox.critical(self.gui,'import BIASD','There was a problem trying to import BIASD. Check the path in preferences >> biasd_path')
			self.gui.log('Failed to import BIASD; check biasd_path preference',True)
			return

		success1,color = self.hmm_get_colorchannel(color)
		if success1:
			try:
				y,ran = self.hmm_get_traces(color)
			except:
				return

		#
		# y = np.concatenate(y)
		# y = y[np.isfinite(y)]
		#
		# e1 = b.distributions.normal(pp['biasd_prior_e1_m'],pp['biasd_prior_e1_s'])
		# e2 = b.distributions.normal(pp['biasd_prior_e2_m'],pp['biasd_prior_e2_s'])
		# sigma = b.distributions.uniform(pp['biasd_prior_sig_l'],pp['biasd_prior_sig_u'])
		# k1 = b.distributions.gamma(pp['biasd_prior_k1_a'],pp['biasd_prior_k1_b'])
		# k2 = b.distributions.gamma(pp['biasd_prior_k2_a'],pp['biasd_prior_k2_b'])
		# priors = b.distributions.parameter_collection(e1, e2, sigma, k1, k2)
		# print b.laplace.laplace_approximation(y, priors, self.gui.prefs['tau'], nrestarts=4, verbose=False, threads=pp['ncpu'], device=0)
		QMessageBox.critical(self.gui,'BIASD','BIASD: We are not really doing this  yet... What do you want?')

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
