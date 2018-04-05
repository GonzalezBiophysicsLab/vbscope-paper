import numpy as np

def fake_data(tmax=5000,outliers = False):
	## tmax is an integer

	## reproducable
	np.random.seed(20180404)

	tmat = np.array(((.985,.015),(.005,.995)))
	state = 0
	m = np.array((-1,1.))
	s = np.array((2.,2.))*.25

	d = np.empty((tmax))
	for i in range(d.shape[0]):
		if np.random.rand() > tmat[state,state]:
			if state == 0:
				state = 1
			elif state == 1:
				state = 0
		d[i] = np.random.normal(scale=s[state]) + m[state]

	if outliers:
		xx = np.random.randint(low=0,high=d.size/10,size=1)
		x = np.random.randint(low=0,high=d.size,size=xx)
		d[x] = np.random.uniform(-10,10,size=xx)

	t = np.arange(d.size)

	return t,d
