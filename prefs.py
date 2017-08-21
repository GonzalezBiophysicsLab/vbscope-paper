## preferences
import numpy as np
import multiprocessing as mp

default = {

'ncolors':2,

'wavelengths_nm':np.array((570.,680.,488.,800.)),

'pixel_size':13300,
'magnification':60,
'binning':2,
'numerical_aperture':1.2,

'nsearch':3,
'nintegrate':5,
'clip border':3,

'color map':'Greys_r',#'viridis',

'tau':0.1,
'bleedthrough':0.05,
'same_cutoff':1.,

'playback_fps':100,

'alignment_order':4,
'contrast_scale':10.,

'maxiterations':1000,
'threshold':1e-10,
'ncpu':mp.cpu_count(),
'nstates':4,

'downsample':1,
'snr_threshold':.5,
'pb_length':10,

'plotter_xmin':0,
'plotter_xmax':-1,
'plotter_n_xbins':41,
'plotter_n_ybins':41,
'plotter_floor':1.0,
'plotter_n_levels':50,
'plotter_smoothx':5.,
'plotter_smoothy':1.

}
