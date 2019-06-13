import numpy as np
import numba as nb
from numba import cuda

################################################################################

@nb.jit(nopython=True,nogil=True,parallel=True)
def movie_div_trace(m,p):
	out = np.zeros_like(m)
	for i in nb.prange(m.shape[0]):
		for j in range(m.shape[1]):
			for k in range(m.shape[2]):
				out[i,j,k] = m[i,j,k] / p[i]
		# out[i] = m[i] / p[i,None,None]
	return out

@nb.jit(nopython=True,nogil=True,parallel=True)
def movie_div_const(m,p):
	out = np.zeros_like(m)
	for i in nb.prange(m.shape[0]):
		for j in range(m.shape[1]):
			for k in range(m.shape[2]):
				out[i,j,k] = m[i,j,k] / p
	return out

@nb.jit(nopython=True,nogil=True,parallel=True)
def movie_div_image(m,p):
	out = np.zeros_like(m)
	for i in nb.prange(m.shape[0]):
		for j in range(m.shape[1]):
			for k in range(m.shape[2]):
				out[i,j,k] = m[i,j,k] / p[j,k]
	return out

@nb.jit(nopython=True,nogil=True,parallel=True)
def spatial_median_filter(m,n):
	out = np.zeros_like(m)
	for i in nb.prange(m.shape[0]):
		for j in range(m.shape[1]):
			xmin = max(0,i-n)
			ymin = max(0,j-n)
			xmax = min(m.shape[0],i+n)
			ymax = min(m.shape[1],j+n)

			out[i,j] = np.median(m[xmin:xmax,ymin:ymax])
	return out

from scipy.ndimage import uniform_filter
def spatial_mean_filter(m,n):
	if m.ndim == 3:
		return uniform_filter(m,[0,n,n])
	return uniform_filter(m,n)

@nb.jit(nopython=True,nogil=True,parallel=True)
def time_median_over_movie(m):
	out = np.zeros((m.shape[1],m.shape[2]))
	for i in nb.prange(m.shape[1]):
		for j in range(m.shape[2]):
			out[i,j] = np.median(m[:,i,j])
	return out

@nb.jit(nopython=True,nogil=True,parallel=True)
def spatial_median_over_movie(m):
	out = np.zeros(m.shape[0])
	for i in nb.prange(m.shape[0]):
		out[i] = np.median(m[i])
	return out

@nb.jit(nopython=True,nogil=True,parallel=True)
def spatial_mean_over_movie(m):
	out = np.zeros(m.shape[0])
	for i in nb.prange(m.shape[0]):
		out[i] = np.mean(m[i])
	return out

@nb.jit(nopython=True,nogil=True,parallel=True)
def subtract_movie_with_delay(a,n):
	out = np.zeros_like(a)
 	for i in nb.prange(a.shape[0]-n):
		out[i] = a[n+i] - a[i]
	return out

################################################################################
try:
	@cuda.jit("void(uint16[:,:,:],float32[:,:],float32[:,:],int32,int32)")
	def _kernel_calc_avgs(movie,num,denom,frame,nframes):
		i, j = cuda.grid(2)
		if i < movie.shape[1] and j < movie.shape[2]:
			fmin = max(0,frame-nframes)
			fmax = min(movie.shape[0],frame+nframes)
			denom[i,j] = 0
			num[i,j] = 0
			for k in range(fmin,frame):
				denom[i,j] += movie[k,i,j]
			denom[i,j] /= (frame-fmin)
			for k in range(frame,fmax):
				num[i,j] += movie[k,i,j]
			num[i,j] /= (fmax-frame)

	@cuda.jit("void(float32[:,:], float32[:,:], float32[:,:],float32[:])")
	def _kernel_mean2d(input1,input2,scratch,output):
		## Assume input1 and input2 are the same shape as output
		## calculates the average of input1 and input2 and stores in output[0,:1]

		tx = cuda.threadIdx.x
		ty = cuda.blockIdx.x
		bw = cuda.blockDim.x
		i = tx + ty * bw

		if i < scratch.shape[0]:
			scratch[i,0] = 0
			for j in range(scratch.shape[1]):
				scratch[i,0] += input1[i,j] ## sum  along 2nd dimension
				scratch[i,1] += input2[i,j]
			cuda.syncthreads()

			if i == scratch.shape[0] - 1:
				for k in range(2):
					output[k] = 0.
					for j in range(scratch.shape[0]):
						output[k] += scratch[j,k] ## add summed dimension
						scratch[j,k] = 0 ## Reset values
					output[k] /= scratch.size ## calc average

	@cuda.jit("void( float32[:,:],float32[:,:],float32[:,:],float32[:])")
	def _kernel_ratiometric(num,denom,out,power):
		i, j = cuda.grid(2)
		if i < out.shape[0] and j < out.shape[1]:
			out[i,j] = (num[i,j]/power[0]) / (denom[i,j]/power[1])

	def setup_ratiometric(movie):
		nt,nx,ny = movie.shape

		threads = 32
		block_dim = (threads,threads)
		grid_dim = (nx/block_dim[0]+1, ny/block_dim[1]+1)

		stream = cuda.stream()

		d_movie = cuda.to_device(movie, stream)          # to device and don't come back
		return [d_movie,stream,block_dim,grid_dim]

	def calc_ratiometric_frame(frame,nframes,cuda_stuff):

		d_movie,stream,block_dim,grid_dim = cuda_stuff

		nt,nx,ny = d_movie.shape
		out = np.zeros((nx,ny),dtype='float32')

		d_out = cuda.to_device(out, stream)
		d_num = cuda.to_device(np.zeros_like(out),stream)
		d_denom = cuda.to_device(np.zeros_like(out),stream)
		d_power = cuda.to_device(np.zeros(2,dtype='float32'),stream)

		_kernel_calc_avgs[grid_dim,block_dim,stream](d_movie,d_num,d_denom,frame,nframes)

		threads = 256
		blocks = d_out.shape[0]/threads + 1
		_kernel_mean2d[blocks,threads,stream](d_num,d_denom,d_out,d_power)

		_kernel_ratiometric[grid_dim,block_dim,stream](d_num,d_denom,d_out,d_power)

		d_out.to_host(stream)
		stream.synchronize()
		return out

	def ratiometric_imaging(movie,nframes=10):
		from tqdm import tqdm
		cuda_stuff = setup_ratiometric(movie)
		out = np.empty_like(movie,dtype='float32')
		for i in tqdm(list(range(movie.shape[0]))):
			out[i] = calc_ratiometric_frame(i,nframes,cuda_stuff)
		return out

################################################################################

	from math import floor
	@cuda.jit
	def _kernel_camera(rng_states, nt, out):
		x,y = cuda.grid(2)
		if x < out.shape[1] and y < out.shape[2]:
			for t in range(out.shape[0]):
				z = xoroshiro128p_uniform_float32(rng_states,x*out.shape[1]+y)
				out[t,x,y] = int(floor(z*4096))*16

except:
	print("No GPU")


################################################################################

def flat_field(m,bg):
	if bg.ndim == 3:
		bg = time_median_over_movie(m)
	return movie_div_image(m,bg)

@nb.jit(nopython=True,nogil=True,parallel=True)
def pseudo_flat_field(m,kernelsize=51):
	for i in range(m.shape[0]):
		m[i] = m[i] / spatial_median_filter(m[i],kernelsize)
	return m

@nb.jit(nopython=True,nogil=True,parallel=True)
def dynamic_imaging(m):
	ref = time_median_over_movie(m)
	for i in range(m.shape[0]):
		m[i] -= ref
	return m

@nb.jit(nopython=True,nogil=True,parallel=True)
def differential_imaging(m,lagframes=100):
	return subtract_movie_with_delay(m,lagframes)

def generate_noise_movie(nt,nx,ny):
	try:
		if nx*ny*nt*2 > 8000000000:
			raise Exception("Not enough memory")

		gpu = cuda.get_current_device()
		tpb = int(np.sqrt(gpu.MAX_THREADS_PER_BLOCK))
		msize = (nt,nx,ny)
		block_dim = (tpb,tpb)
		grid_dim = (nx/block_dim[0]+1, ny/block_dim[1]+1)

		rng_states = create_xoroshiro128p_states(msize[1]*msize[2],seed=1)
		out = np.zeros(msize,dtype='uint16')

		_kernel_camera[grid_dim,block_dim](rng_states,msize[0],out)
	except:
		out =  np.random.randint(65535,size=(1000,128,128),dtype='uint16')
	return out.astype('float32')

################################################################################

@nb.jit(nopython=True,nogil=True,parallel=True)
def normalize_movie_power(m):
	power = spatial_mean_over_movie(m)
	power /= np.median(power)
	m = movie_div_trace(m,power)
	return m

def bin2x(movie):
	return (movie[:,::2,::2]/4 + movie[:,1::2,::2]/4 + movie[:,::2,1::2]/4 + movie[:,1::2,1::2]/4)

################################################################################

def _test():
	import time
	print('start')
	t0 = time.time()
	movie = generate_noise_movie(1000,512,512)
	movie = bin2x(movie)
	print(movie.shape)
	t1 = time.time()
	print("Generate Movie: %f gb in %f sec"%(movie.size*2/1000000000.,t1-t0))

	t0 = time.time()
	flat_field(movie,movie[:100])
	t1=time.time()
	print("Flat Field:",t1-t0)
	t0 = time.time()
	pseudo_flat_field(movie,11)
	t1=time.time()
	print("Pseudo Flat Field:",t1-t0)
	t0 = time.time()
	dynamic_imaging(movie)
	t1=time.time()
	print("Dynamic Imaging:",t1-t0)
	t0 = time.time()
	differential_imaging(movie,99)
	t1=time.time()
	print("Differential Imaging:",t1-t0)
	try:
		t0 = time.time()
		ratiometric_imaging(movie,20)
		t1=time.time()
		print("Ratiometric Imaging:",t1-t0)
	except:
		"You probably don't have CUDA for ratiometric imaging"
	t0 = time.time()
	normalize_movie_power(movie)
	t1=time.time()
	print("Normalize Power:",t1-t0)

if False:
	_test()
