# movie loading
import numpy as np

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

	movie = []
	for i in range(frames):
		im.seek(i)
		movie.append(im)
	movie = np.array(movie)

	print "PIL - loaded %d frames"%(movie.shape[0])
	return movie

def io_load_tif(filename):
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

def specify_split_movie(movie, origins, height, width):
	'''
	Input movie(T,X,Y)
	Returns movie(T,X',Y',D)
	'''
	# Time, X, Y, Regions
	splitted = np.empty((movie.shape[0],height,width,np.shape(origins)[0]))
	for i in range(splitted.shape[3]):
		x0,y0 = origins[i]
		splitted[...,i] = movie[:,x0:x0+height,y0:y0+width]
	return splitted

def auto_split_movie(movie,ncolors=2):
	'''
	Use ncolors = 1, 2, 3, or 4. Note, 3 and 4 both give 4 colors

	Input movie(T,X,Y)
	Returns movie(T,X',Y',D)
	'''
	t,d1,d2 = movie.shape
	if ncolors == 1:
		return movie[...,None]
	elif ncolors == 2:
		# Left, Right
		origins = np.array(((0,0),(0,d2/2)))
		return specify_split_movie(movie,origins,d1,d2/2)
	elif ncolors == 4 or ncolors == 3:
		# Clockwise????? from upper left????
		origins = np.array((
			(0   ,    0),
			(0   , d2/2),
			(d1/2, d2/2),
			(d1/2,    0)
		))
		return specify_split_movie(movie,origins,d1/2,d2/2)
