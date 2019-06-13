## kernel sample
from scipy.stats import gaussian_kde

def kernel_sample(x,nstates):
	kernel = gaussian_kde(x)
	m = kernel.resample(nstates).flatten()
	m.sort()
	return m
