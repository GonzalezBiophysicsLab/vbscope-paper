import matplotlib.pyplot as plt
import numpy as np


class data_container():
	def __init__(self,parent=None):
		self.flag_movie = False

		self.movie = None
		self.background = None

		self.filename = None

		self.current_frame = 0
		self.total_frames = 0

		self.dxy = np.array((0,0))

		self.image_contrast = np.array((0.,100.))

		self.spots = None

		self.ncolors = 2
		self.transforms = [None]

		self.background = None
		self.metadata = []


	def load(self,filename):
		try:
			if filename.endswith('.npy'):
				data = np.load(filename).astype('uint16')
			elif filename.endswith('.tif') or filename.endswith('.stk'):
				from skimage.io import imread
				data = imread(filename,plugin='tifffile')
				# from ..external import tifffile
				# m = tifffile.TiffFile(filename)
				# data = m.asarray().astype(dtype='uint16')

			self.__init__()

			self.flag_movie = True

			if data.ndim == 2:
				data = data[None,...]

			self.movie = data
			self.filename = filename

			self.current_frame = 0
			self.total_frames = self.movie.shape[0]

			self.dxy = self.movie.shape[1:]
			self.background = np.zeros(self.dxy,dtype='i')
		except:
			return False

		try:
			self.metadata = []
			if m.is_micromanager:
				self.metadata.append([filename,m.micromanager_metadata])
			for i in range(len(m.pages)):
				mm = m.pages[i]
				if mm.is_micromanager:
					self.metadata.append(["Page %d"%(i),mm.tags['micromanager_metadata'].value])
		except:
			pass

		return True


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
				[[0,dy],[0,dx//2]],
				[[0,dy],[dx//2,dx]]
			]
			shifts = [
				[0,0],
				[0,dx//2]
			]
		elif self.ncolors == 3:
			r = [
				[[0,dy//2],[0,dx//2]],
				[[0,dy//2],[dx//2,dx]],
				[[dy//2,dy],[dx//2,dx]]
			]
			shifts = [
				[0,0],
				[0,dx//2],
				[dy//2,dx//2]
			]
		elif self.ncolors == 4:
			r = [
				[[0,dy//2],[0,dx//2]],
				[[0,dy//2],[dx//2,dx]],
				[[dy//2,dy],[dx//2,dx]],
				[[dy//2,dy],[dx//2,dx]]
			]
			shifts = [
				[0,0],
				[0,dx//2],
				[dy//2,dx//2],
				[dy//2,0]
			]
		return np.array(r),np.array(shifts)
