## test
from .fxns.fake_data import fake_data
import numpy as np
import matplotlib.pyplot as plt
import time

def test_ml_em_gmm():
	from .ml_em_gmm import ml_em_gmm
	t,d = fake_data(outliers=True)

	nstates = 5
	o = ml_em_gmm(d,nstates)

	cm = plt.cm.jet
	for i in range(nstates):
		c = cm(float(i)/nstates)
		# print i,c
		xcut = o.r.argmax(1) == i
		plt.plot(t[xcut],d[xcut],'o',color=c,alpha=.5)
	plt.show()


def test_ml_em_hmm():
	from .ml_em_hmm import ml_em_hmm

	t,d = fake_data()

	nstates = 2
	o = ml_em_hmm(d[:10],nstates)
	print('hi')
	t0 = time.time()
	o = ml_em_hmm(d,nstates)
	t1 = time.time()
	print(t1-t0)
	print(o.tmatrix)
	print(o.mu)
	print(o.var**.5)
	print(o.r.shape)

	cm = plt.cm.jet
	for i in range(nstates):
		c = cm(float(i)/nstates)
		# print i,c
		xcut = o.r.argmax(1) == i
		plt.plot(t[xcut],d[xcut],'o',color=c,alpha=.5)
		# plt.axhline(y=o.mu[i], color=c)
	plt.plot(t,o.mu[o.viterbi])
	plt.show()

def test_vb_em_gmm():
	from .vb_em_gmm import vb_em_gmm
	t,d = fake_data()

	nstates = 3
	o = vb_em_gmm(d,nstates)

	# for nstates in range(1,5):
		# o = vb_em_gmm(d,nstates)
		# print nstates,o.likelihood[-1]

	# import matplotlib.pyplot as plt
	# cm = plt.cm.jet
	# hy,hx = plt.hist(d,bins=300,histtype='stepfilled',alpha=.5)[:2]
	# for i in range(nstates):
	# 	c = cm(float(i)/nstates)
	# 	# print i,c
	# 	xcut = o.r.argmax(1) == i
	# 	plt.hist(d[xcut],bins=hx,color=c,alpha=.5)
	# plt.show()
	cm = plt.cm.jet
	f,a = plt.subplots(1)
	hy,hx=a.hist(d,bins=100,histtype='stepfilled',alpha=.5,density=True)[:2]
	x = np.linspace(hx[0],hx[-1],10000)
	ytot = np.zeros_like(x)
	def pp_normal(x,m,v):
		return np.exp(-.5*np.log(2.*np.pi) -.5*np.log(v) - .5/v*(x-m)**2.)

	for i in range(nstates):
		c = cm(float(i)/nstates)
		# print i,c
		# xcut = o.r.argmax(1) == i
		y = o.ppi[i]*pp_normal(x,np.array((o.mu[i])),np.array((o.var[i])))
		a.plot(x,y,lw=1,label='%d'%(i))
		ytot+=y
		# a[0].plot(t[xcut],d[xcut],'o',color=c,alpha=.5)
	a.plot(x,ytot,lw=1.5,color='k')
	a.legend()
	# for i in range(o.likelihood.shape[1]):
		# a[1].plot(o.likelihood[:,i],label='%d'%(i))
	# a[1].legend()
	plt.show()


def test_vb_em_hmm():

	from .vb_em_hmm import vb_em_hmm
	t,d = fake_data()

	# nstates = 2
	lls = []
	for nstates in range(1,11):
		o = vb_em_hmm(d,nstates)
		print(nstates)
		lls.append(o.likelihood)
	l = [ll[-1,0] for ll in lls]
	plt.figure()
	plt.plot(list(range(1,11)),l,'-o')
	plt.show()
	#
	# print o.tmatrix
	# print o.mu
	# print o.var**.5
	# print o.r.shape
	#
	# import matplotlib.pyplot as plt
	cm = plt.cm.jet

	f,a = plt.subplots(1)
	for i in range(nstates):
		c = cm(float(i)/nstates)
		# print i,c
		xcut = o.viterbi==i
		a.plot(t[xcut],d[xcut],'o',color=c,alpha=.5)
		# plt.axhline(y=o.mu[i], color=c)
	a.plot(t,o.mu[o.viterbi])

	# a[1] = plt.plot(o.likelihood[:,0],'k')
	plt.show()

def test_consensus_vb_em_hmm():

	from .consensus_vb_em_hmm import consensus_vb_em_hmm,consensus_vb_em_hmm_parallel
	t,d = fake_data()
	tt,dd = fake_data()
	ddd = [d,dd[:-10]]

	# nstates = 2
	lls = []
	for nstates in range(1,5):
		o = consensus_vb_em_hmm_parallel(ddd,nstates,nrestarts=4,ncpu=4)
		# o = consensus_vb_em_hmm(ddd,nstates)
		print(nstates)
		lls.append(o.likelihood)
	l = [ll[-1,0] for ll in lls]
	plt.figure()
	plt.plot(list(range(1,len(lls)+1)),l,'-o')
	plt.show()
	#
	# print o.tmatrix
	# print o.mu
	# print o.var**.5
	# print o.r.shape
	#
	# import matplotlib.pyplot as plt
	cm = plt.cm.jet
	o = consensus_vb_em_hmm_parallel(ddd,2,nrestarts=4,ncpu=4)

	f,a = plt.subplots(1)
	for i in range(nstates):
		c = cm(float(i)/nstates)
		# print i,c
		xcut = o.viterbi[0]==i
		a.plot(t[xcut],d[xcut],'o',color=c,alpha=.5)
		# plt.axhline(y=o.mu[i], color=c)
	a.plot(t,o.mu[o.viterbi[0]])

	# a[1] = plt.plot(o.likelihood[:,0],'k')
	plt.show()

# test_ml_em_gmm()
# test_ml_em_hmm()
# test_vb_em_gmm()
# test_vb_em_hmm()
test_consensus_vb_em_hmm()
