import numpy as np
import ctypes
from sys import platform
import os

path = os.path.dirname(__file__) + '/sm_ssa'

if platform == 'darwin':
	_sopath = path+'/sm_ssa-mac'
elif platform == 'linux' or platform == 'linux2':
	_sopath = path + '/sm_ssa-linux'
print _sopath
_lib = np.ctypeslib.load_library(_sopath, '.')

_lib.sm_ssa.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_int, np.ctypeslib.ndpointer(dtype = np.double), np.ctypeslib.ndpointer(dtype = np.int32), np.ctypeslib.ndpointer(dtype = np.double), np.ctypeslib.ndpointer(dtype=np.int32)]
_lib.sm_ssa.restype  = ctypes.c_void_p


_lib.render_trace.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, np.ctypeslib.ndpointer(dtype = np.double),  np.ctypeslib.ndpointer(dtype = np.double), np.ctypeslib.ndpointer(dtype = np.int32), np.ctypeslib.ndpointer(dtype = np.double), np.ctypeslib.ndpointer(dtype=np.double), np.ctypeslib.ndpointer(dtype=np.double)]
_lib.render_trace.restype  = ctypes.c_void_p

class simtrace:
	@staticmethod
	def generate_states(rates,tlength):
		from scipy.linalg import expm
		kk = rates.copy()[:-1,:-1]
		for i in range(kk.shape[0]):
			kk[i,i] = - kk[i].sum()
		pst = expm(kk*100000)[0].cumsum()
		np.random.seed()
		p = np.random.rand()
		initialstate = np.searchsorted(pst,p)

		nstates = rates.shape[0]
		rates = rates.flatten()
		n = np.max((int(np.floor(tlength*rates.max())),1))*2

		# initialstate = np.random.randint(nstates)

		states = np.zeros(n, dtype=np.int32)
		dwells = np.zeros(n, dtype=np.double)
		cut = np.array(0,dtype=np.int32)
		_lib.sm_ssa(n, (tlength), nstates, initialstate, rates, states, dwells,cut)
		states = states[:cut]
		dwells = dwells[:cut]
		if np.size(states) == 0:
			states = [initialstate]
			dwells = [tlength]

		return np.array((states,dwells))

	@staticmethod
	def render_trace(trace,steps,dt,emission):
		states = trace[0].astype(np.int32)
		dwells = trace[1]
		times = dwells.cumsum().astype(np.double)
		times = np.append(0,times)
		timesteps = states.shape[0]
		steps = int(steps)

		nstates = emission.size
		emissions = emission.astype(np.double)
		x = np.arange(steps,dtype=np.double)*dt + dt
		y  = np.zeros_like(x)
		if timesteps == 0:
			print trace
		_lib.render_trace(steps, timesteps, nstates, x, y, states, times, dwells, emissions)
		return x,y

	def __init__(self,rates,emissions,noise,frames,tau):
		self.k = rates
		self.emission = emissions
		self.dt = tau
		self.sigma = noise
		self.simulate(frames)


	def simulate(self,frames):
		np.random.seed()
		self.a = self.generate_states(self.k,frames*self.dt)
		self.x,self.y = self.render_trace(self.a,frames,self.dt,self.emission)
		self.raw = self.y.copy()
		self.y += np.random.normal(scale=self.sigma,size=self.y.size)

	def simulate_fret(self,frames,da_sum):
		np.random.seed()
		self.a = self.generate_states(self.k,frames*self.dt)
		self.x,self.y = self.render_trace(self.a,frames,self.dt,self.emission)
		noise = np.random.normal(scale=self.sigma,size=self.y.size)

		self.y2 = da_sum*(noise+self.y)#(self.y + np.random.normal(scale=self.sigma,size=self.y.size))
		self.y1 = da_sum*(1.-self.y-noise)#(1.-(self.y+np.random.normal(scale=self.sigma,size=self.y.size)))
		self.y1[self.y == 0] = da_sum*np.random.normal(scale=self.sigma,size=(self.y==0).sum())
		# self.y1[self.y1 < 0] = 0
		# self.y2[self.y2 < 0] = 0
		#self.y1 = np.random.poisson(lam=self.y1).astype('f')
		#self.y2 = np.random.poisson(lam=self.y2).astype('f')
