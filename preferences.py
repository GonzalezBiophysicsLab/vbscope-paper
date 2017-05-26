## preferences
import numpy as np
import multiprocessing as mp

default = {

'filename':'./',

'image_dimensions':np.array((512,512)),
'crop':np.array(((0,-1),(0,-1))),
'ncolors':2,

'wavelengths_nm':np.array((570.,680.)),

'pixel_size':13300,
'magnification':60,
'numerical_aperture':1.2,

'nsearch_min':5,
'nsearch_max':3,

'maxiterations':1000,
'threshold':1e-10,
'nrestarts':10,
'ncpu':mp.cpu_count(),

'tau':0.5,
'playback_fps':25,
'image_contrast':np.array((0.,100.)),
'contrast_scale':10.

}
