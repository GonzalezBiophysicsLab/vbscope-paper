## Install the Required Python Libraries
pyqt 5.6
numpy 1.13
matplotlib 2.0
scipy 0.19
scikit-image 0.13
scikit-learn 0.19.0
numba 0.34
gitpython 2.1

### Using conda
``` bash
conda install scikit-image scipy numpy matplotlib pyqt scikit-learn numba gitpython
```

## Make a Terminal Shortcut
To your rc file (eg ~/.bashrc, ~/.zshrc, ~/.profile, etc) add
``` bash
alias vbscope="python /home/username/directory/to/vbscope/gui.py"
alias vbscope_plot="python /home/username/directory/to/vbscope/docks/plotter.py"
```

## Parameters
Explanations of the parameters in the preferences dock:

### Experiment Parameters
* tau - the frame period in seconds
* pixel_size - length of the (square) pixels on the camera in nanometers. Andor 888 is 13300, Andor 897 is 16000.
* numerical_aperture - the NA of the objective used to the collect the movie
* channel_wavelengths - wavelengths of each color channel. This is used the calculate the PSF width
* binning - factor of binning in the microscope image. Used to calculate the PSF width
* magnification - Magnification of objective used in microscope
* bleedthrough - fraction of donor intensity removed from acceptor intensity in the plots and exported traces. Multiple colors is complicated. The first `n` (number of colors) entries correspond to the fraction of color 1 to remove from colors 1 through n, respectively. The next `n` entries correspond to the fraction of color 2 to remove from colors 1 through n, respectively... and so on.

### Display Parameters
* channel_colors - the color used to plot each channel (string colors accepted)
* color map - Matplotlib colormap for the movie image
* contast_scale - multiplier for determining non-linearity of contrast sliders. Higher is  a wider range
* playback_fps - upper-limit of frames per second for playing movie

### Spotfinding Parameters
* nsearch - number of pixels (in one dimension) that are used for finding a local min or local max. Can only be odd numbers
* ncpu - number of cpus to use in parallel when spotfinding in multiple frames
* threshold - relative threshold the lowerbound must decrease by during VBEM in spotfinding for the algorithm to converge
* nstates - number of spot classes to use during spotfinding
* maxiterations - maximum number of VBEM iterations to run when spotfinding
* clip border - number of pixels to exclude from each border edge when spot finding

### Trace Extraction Parameters
* same_cutoff - distance cutoff in pixels of which spots are the same spot in different color channels
* nintegrate - number of pixels (in one dimension) that are used for integrating the PSF to get the intensity values

### Alignment Parameters
* alignment_order - order of the polynomial used for alignment when loaded in alignment file

### Plotting Parameters
* downsample - number of frames to sum together into one datapoint. Time of this new datapoint is the time of the first datapoint in the sum.
* plotter_xmin - first frame to use (in frames) for ensemble plots
* plotter_xmax - last frame to use (in frames) for ensemble plots. -1 is last
* plotter_n_xbins - number of histogram bins for x axis for all ensemble plots
* plotter_n_ybins - number of histogram bins for y axis for all ensemble plots
* plotter_floor - don't display less than this value in 2D heat maps. For 2D FRET, this should be between 0 and 1. For the TDP, this should be greater than 1
* plotter_n_levels - number of filled contour levels for 2D heat maps
* plotter_smoothx - standard deviation of gaussian (in datapoints) used to smooth 2D heat maps in the x direction
* plotter_smoothy - standard deviation of gaussian (in datapoints) used to smooth 2D heat maps in the y direction

### Plot Processing Parameters
* snr_threshold - traces with estimated SNR less than this number can be removed
* pb_length - number of datapoints for sliding window when calculating variance for photobleaching variance method. Also the minimum number of frames between pre-bleach and post-bleach points for a trace that can be kept
