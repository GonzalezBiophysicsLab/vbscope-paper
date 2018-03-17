from PyQt5.QtWidgets import QInputDialog,QFileDialog

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

	## Remove trajectories with number of kept-frames < threshold
	def cull_pb(self):
		if not self.d is None and self.gui.ncolors == 2:
			self.safe_hmm()
			pbt = self.pb_list.copy()
			pret = self.pre_list.copy()
			dt = pbt-pret
			cut = dt > self.gui.prefs['pb_length']

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

	def cull_photons(self):
		if not self.d is None:
			combos = ['%d'%(i) for i in range(self.gui.ncolors)]
			combos.append('0+1')
			c,success1 = QInputDialog.getItem(self.gui,"Color","Choose Color channel",combos,editable=False)
			if success1:
				self.safe_hmm()
				keep = np.zeros(self.d.shape[0],dtype='bool')
				y = np.zeros(self.d.shape[0])
				for ind in range(self.d.shape[0]):
					intensities = self.d[ind].copy()
					bts = self.gui.prefs['bleedthrough'].reshape((4,4))
					for i in range(self.gui.ncolors):
						for j in range(self.gui.ncolors):
							intensities[j] -= bts[i,j]*intensities[i]

					for i in range(self.gui.ncolors):
						intensities[i] = self.gui.prefs['convert_c_lambda'][i]/self.gui.prefs['convert_em_gain']*intensities[i]
					cc = c.split('+')
					for ccc in cc:
						y[ind] += intensities[int(ccc)].sum()
				# x = np.logspace(0,np.log10(y.max()),1000)
				x = np.linspace(y.min(),y.max(),10000)
				surv = np.array([(y > x[i]).sum()/float(y.size) for i in range(x.size)])

				# if self.docks['plots_surv'][0].isHidden():
				# 	self.docks['plots_surv'][0].show()
				# self.docks['plots_surv'][0].raise_()
				# popplot = self.docks['plots_surv'][1]
				# popplot.ax.cla()
                #
				# # plt.semilogx(x,surv)
				# popplot.ax.plot(x,surv)
				# popplot.ax.set_ylim(0,1)
				# popplot.ax.set_ylabel('Survival Probability')
				# popplot.ax.set_xlabel('Total Number of Photons')
				# popplot.f.tight_layout()
				# popplot.f.canvas.draw()


				threshold,success2 = QInputDialog.getDouble(self.gui,"Photon Cutoff","Total number of photons required to keep a trajectory",value=1000.,min=0.,max=1e10,decimals=3)
 				if success2:
					keep = y > threshold
					d = self.d[keep]
					self.gui.plot.index = 0
					self.gui.initialize_data(d)
					self.gui.plot.initialize_plots()
					self.gui.initialize_sliders()
					self.gui.log('Cull Photons: %d traces with less than %d total photons in channel %s removed'%((keep==0).sum(),threshold,c),True)
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

		if self.gui.pb_remove_check.isChecked() and self.gui.ncolors == 2:
			from ..supporting.photobleaching import remove_pb_all
			fpb[0] = remove_pb_all(fpb[0])
		return fpb

	def get_viterbi_data(self):
		if not self.hmm_result is None:
			v = np.empty_like(self.fret[0]) + np.nan
			for i in range(v.shape[0]):
				if self.hmm_result.ran.count(i) > 0:
					ii = self.hmm_result.ran.index(i)
					v[i,self.pre_list[i]:self.pb_list[i]] = self.hmm_result.viterbi[ii]

			# if self.gui.prefs['synchronize_start_flag'] != 'True':
			# 	v[i,:self.gui.prefs['plotter_min_time']] = np.nan
			# 	v[i,self.gui.prefs['plotter_max_time']:] = np.nan

			checked = self.gui.classes_get_checked()
			v = v[checked]
			return v
		else:
			return None

	def run_hmm(self):
		from ..supporting import simul_vbem_hmm as hmm

		if not self.d is None and self.gui.ncolors == 2:
			if self.gui.prefs['hmm_binding_expt'] == 'True':
				nstates = 2
				success = True
			else:
				nstates,success = QInputDialog.getInt(self.gui,"Number of HMM States","Number of HMM States",min=2)
			if success and nstates > 1:
				self.update_fret()
				y = []
				checked = self.gui.classes_get_checked()
				ran = []
				if self.gui.prefs['hmm_binding_expt'] == 'True':
					z = self.get_fluor().sum(1)
				for i in range(self.fret.shape[1]):
					if checked[i]:
						if self.gui.prefs['hmm_binding_expt'] == 'True':
							yy = z[i,self.pre_list[i]:self.pb_list[i]]
						else:
							yy = self.fret[0,i,self.pre_list[i]:self.pb_list[i]]
							yy[np.isnan(yy)] = -1.
							yy[yy < -1.] = -1.
							yy[yy > 2] = 2.
						if yy.size > 5:
							y.append(yy)
							ran.append(i)
				nrestarts = self.gui.prefs['hmm_nrestarts']
				priors = [hmm.initialize_priors(y,nstates,flag_vbfret=False,flag_custom=True) for _ in range(nrestarts)]
				if self.gui.prefs['hmm_binding_expt'] == 'True':
					for iii in range(nrestarts):
						priors[iii][0] = np.array((0,1000.)) ## m
						priors[iii][1] = np.ones(2) ## beta

				if self.gui.prefs['hmm_sigmasmooth'] == "True":
					sigma_smooth = 0.5
				else:
					sigma_smooth = False
				result,lbs = hmm.hmm(y,nstates,priors,nrestarts,sigma_smooth)
				ppi = np.sum([result.gamma[i].sum(0) for i in range(len(result.gamma))],axis=0)
				ppi /= ppi.sum()

				report = "%s\nHMM - k = %d, iter= %d, lowerbound=%f"%(lbs,nstates,result.iterations,result.lowerbound)
				report += '\n    f: %s'%(ppi)
				report += '\n    m: %s'%(result.m)
				report += '\nm_sig: %s'%(1./np.sqrt(result.beta))
				report += '\n  sig: %s'%((result.b/result.a)**.5)
				rates = -np.log(1.-result.Astar)/self.gui.prefs['tau']
				for i in range(rates.shape[0]):
					rates[i,i] = 0.
				report += '\n   k:\n'
				report += '%s'%(rates)
				self.gui.log("HMM report - %d states"%(nstates),True)
				self.gui.log(report)
				self.hmm_result = result
				self.hmm_result.ran = ran
				self.gui.plot.initialize_hmm_plot()
				self.gui.plot.update_plots()


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

				if self.gui.prefs['hmm_binding_expt'] == 'True':
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


	def remove_beginning(self):
		if not self.d is None:
			self.safe_hmm()
			nd,success = QInputDialog.getInt(self.gui,"Remove Datapoints","Number of datapoints to remove starting from the beginning of the movie")
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
			# l1 = calc_pb_time(self.fret,self.gui.prefs['pb_length_cutoff'])
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
		if self.gui.prefs['hmm_bound_dynamics'] == 'True':
			return q[:,1,:]
		else:
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
			keep = self.deadprob > threshold
			self.gui.plot.index = 0
			self.gui.initialize_data(self.d[keep])
			self.gui.plot.initialize_plots()
			self.gui.initialize_sliders()
			self.gui.log('Remove Dead Frames: %d traces with less than %d total frames in the summed channel removed'%((keep==0).sum(),threshold),True)
			self.gui.update_display_traces()

def normal(x,m,v):
	return 1./np.sqrt(2.*np.pi*v) * np.exp(-.5/v*(x-m)**2.)
