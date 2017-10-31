import matplotlib.pyplot as plt
import numpy as np

class data_container():
	def __init__(self,parent=None):
		self.parent = parent
		self.initialize()

	def initialize(self):
		self.flag_movie = False

		self.movie = None
		self.filename = None
		self.dispname = None
		self.filesize = 0

		self.current_frame = 0
		self.total_frames = 0

		self.dxy = np.array((0,0))

		self.spots = None
		self.crop = np.array(((0,-1),(0,-1)),dtype='i')

		self.ncolors = 2
		self.transforms = [None]

		self.background = None
		self.image_contrast = np.array((0.,100.))


	def load(self,filename):
		data = _io_load_tif(filename)

		if not data is None:
			self.initialize()
			self.flag_movie = True

			if data.ndim == 2:
				data = data[None,...]
			self.movie = data
			self.filename = filename
			self.dispname = '...'+self.filename[-40:]
			from os.path import getsize
			self.filesize = int(np.round(getsize(self.filename)/1e6)) #Mb

			self.current_frame = 0
			self.total_frames = self.movie.shape[0]

			self.dxy = self.movie.shape[1:]
			self.background = np.zeros(self.dxy,dtype='f')

			self.transforms = None

			return True

		return False

	def regions_shifts(self):
		dy,dx = self.dxy # it's backwards...... b/c ploting issues.

		if self.ncolors == 1:
			r = [
				[[0,dy],[0,dx]]
			]
			shifts = [
				[0,0]
			]
		elif self.ncolors == 2:
			r = [
				[[0,dy],[0,dx/2]],
				[[0,dy],[dx/2,dx]]
			]
			shifts = [
				[0,0],
				[0,dx/2]
			]
		elif self.ncolors == 3:
			r = [
				[[0,dy/2],[0,dx/2]],
				[[0,dy/2],[dx/2,dx]],
				[[dy/2,dy],[dx/2,dx]]
			]
			shifts = [
				[0,0],
				[0,dx/2],
				[dy/2,dx/2]
			]
		elif self.ncolors == 4:
			r = [
				[[0,dy/2],[0,dx/2]],
				[[0,dy/2],[dx/2,dx]],
				[[dy/2,dy],[dx/2,dx]],
				[[dy/2,dy],[dx/2,dx]]
			]
			shifts = [
				[0,0],
				[0,dx/2],
				[dy/2,dx/2],
				[dy/2,0]
			]
		return np.array(r),np.array(shifts)


################################################################################

def _pil_load(filename,frames=None):
	#Time, X, Y

	try:
		from PIL import Image
		im = Image.open(filename)
	except IOError:
		print "Cannot open file."
		return None

	if frames is None:
		frames = 1
		try:
			while 1:
				im.seek(frames)
				frames += 1
		except EOFError:
			pass

	nx,ny = im.size
	movie = np.empty((frames,nx,ny))
	for i in range(frames):
		im.seek(i)
		movie[i] = np.array((im.getdata())).reshape((nx,ny))

	print "PIL - loaded %d frames"%(movie.shape[0])
	return movie

def _io_load_tif(filename):
	'''
	Input: string `filename`
	Output: TIF stack as an `np.ndarray` - shape = (T,X,Y)

	Note: Attempts with tifffile (fast), then PIL (slow)
	Failure: returns `None`
	'''

	try:
		import tifffile
		movie = tifffile.TiffFile(filename).asarray()
		print "TIFFFile - loaded %d frames"%(movie.shape[0])
		return movie

	except:
		pass

	return _pil_load(filename)
