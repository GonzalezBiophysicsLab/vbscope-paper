import numpy as np
from simultaneous_vbem_hmm_numba import hmm_with_restarts as hmm
from simultaneous_vbem_hmm_numba import initialize_priors
from check_simulation import plott
from simulate_data import simulate


n = 100
tpb = 2000
l = np.random.exponential(tpb,size=n)
l[l<5.] = 50.
t = []
d = []
q = []
for i in range(n):
	trace = simulate(int(l[i]))
	t.append(trace.x[:-1])
	d.append(trace.y[:-1])
	q.append(trace.raw[:-1])
	np.random.seed()

nstates=6
nrestarts = 4
priors = [initialize_priors(d,nstates,flag_vbfret=False,flag_custom=True) for _ in range(nrestarts)]

result,lbs = hmm(d,nstates,priors,nrestarts)
print result.m

p = plott(t,d,q,result)
