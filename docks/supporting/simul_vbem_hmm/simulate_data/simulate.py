import numpy as np
from sm_sim import simtrace

def simulate(frames):
	# ems = np.array((0.2,0.3,0.5,0.7,0.9))
	# rates = np.array((
	# 	(0.0,1.0,0.0,0.0,0.0),
	# 	(3.0,0.0,2.0,0.0,0.0),
	# 	(0.0,1.5,0.0,4.0,0.0),
	# 	(0.0,0.0,2.5,0.0,1.5),
	# 	(0.0,0.0,0.0,3.0,0.0)
	# )) * 0.01
	# tau = 0.5
	# noise = 0.02

	ems = np.array([0.2,0.3,0.48,0.52,0.7,0.9])
	noise = 0.05
	tau = .05

	ap = .8
	am = .7
	bp = 1.
	bm = .5

	rates = np.array((
		( 0.0,  ap, 0.0, 0.0, 0.0, 0.0),
		(  am, 0.0,  bp,  ap, 0.0, 0.0),
		( 0.0,  bm, 0.0, 0.0,  ap, 0.0),
		( 0.0,  am, 0.0, 0.0,  bp, 0.0),
		( 0.0, 0.0,  am,  bm, 0.0,  bp),
		( 0.0, 0.0, 0.0, 0.0,  bm, 0.0)
	))


	return simtrace(rates,ems,noise,frames,tau)
