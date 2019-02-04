import numpy as np
import numba as nb

# from baseline_filters import gaussian_filter
from .fxns.kmeans import kmeans
from .baseline_updates import update_baseline, update_vn, update_r2, calc_ll

@nb.njit(nb.double[:](nb.double[:],nb.double[:],nb.int64))
def update_model(data,baseline,nstates):
	r,m,v,pi = kmeans(data-baseline,nstates)
	model = np.zeros_like(baseline)
	for i in range(model.size):
		model[i] = m[np.argmax(r[i])]
	return model


@nb.njit(nb.types.Tuple((nb.double[:],nb.double[:]))(nb.double[:],nb.int64))
def kmeans_baseline(data,nstates):
	model = np.zeros_like(data)
	vn = 1./3.*(np.var(data[:20]) + np.var(data[data.size//2-10:data.size//2+11])+ np.var(data[:-20]))
	r2 = (np.var(data) - vn)*6/data.size / vn
	baseline = update_baseline(data,model,r2)
	baseline -= baseline[0]

	lls = np.zeros(1000)
	for i in range(lls.size):
		model = update_model(data,baseline,nstates)
		# baseline = gaussian_filter(update_baseline(data,model,r2),1.)
		baseline = update_baseline(data,model,r2)
		vn = update_vn(data,model,baseline,r2)
		r2 = update_r2(data,model,baseline)
		lls[i] = calc_ll(data,model,baseline,vn,r2)

		# print(i,lls[i],vn,r2)
		if i > 1:
			if np.abs((lls[i] - lls[i-1])/lls[i]) < 1e-10:
				break

	return model,baseline


if __name__ == '__main__':
	## Get data
	# vn = 1.
	# vb = 1.
	# n = 5000
	#
	# import time
	# np.random.seed(int(time.time()))
	# x1 = (np.random.normal(size=n)*np.sqrt(vb)).cumsum()
	# x2 = np.random.normal(size=n)*np.sqrt(vn)
	# data = x1 + x2
	# from fxns.fake_data import fake_data
	# t,d = fake_data(data.size,seed = int(time.time()))
	# data += d*10.
	# data -= data[0]

	data = np.loadtxt('/Users/colin/Desktop/nice.dat')


	model1,baseline1 = kmeans_baseline(data[:8601],2)
	model2,baseline2 = kmeans_baseline(data[8601:],1)
	model = np.concatenate((model1,model2))
	baseline = np.concatenate((baseline1,baseline2))


	import matplotlib.pyplot as plt
	t = np.arange(data.size)*.01

	f,a = plt.subplots(2,2,gridspec_kw={'width_ratios':[9, 1],'wspace':0})
	a[0][0].plot(t,data,lw = 1,alpha=.9)
	a[0][0].plot(t,baseline,lw=1,alpha=.9,zorder=-2)
	a[1][0].plot(t,data-baseline,lw=1,alpha=.9,color='k')
	hy1,hx1 = a[0][1].hist(data,bins=int(np.sqrt(data.size)),orientation='horizontal',histtype='stepfilled',alpha=.8,density=True)[:2]
	hy2,hx2 = a[1][1].hist(data-baseline,bins=int(np.sqrt(data.size)),orientation='horizontal',histtype='stepfilled',alpha=.8,density=True,color='k')[:2]
	# a[1].plot(model)
	# a[2].plot(data-baseline-model)
	for aa in a:
		aa[0].set_xlim(t[0],t[-1])
		aa[1].set_yticks((),())
		aa[1].set_xticks((),())
	a[0][1].set_xlim(0,1.05*hy1.max())
	a[1][1].set_xlim(0,1.05*hy2.max())
	a[1][0].set_xlabel('Time (s)')
	f.tight_layout()
	plt.show()
