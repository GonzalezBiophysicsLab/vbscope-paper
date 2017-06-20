## Install
pyqt 5.6
numpy 1.13
matplotlib 2.0
scipy 0.19
scikit-image 0.13

### Update Python Packages
``` bash
conda install scikit-image scipy numpy matplotlib pyqt
```

### Compile Tifffile
``` bash
python setup.py build_ext --inplace
```

### Terminal Shortcut
To your rc file (eg ~/.bashrc, ~/.zshrc, ~/.profile, etc) add
``` bash
alias vbscope="python /home/username/directory/to/vbscope/gui.py"
```

## Parameters
Explanations of the parameters in the preferences dock:

* nsearch - number of pixels (in one dimension) that are used for finding a local min or local max. Can only be odd numbers
* color map - Matplotlib colormap for the movie image
* clip border - number of pixels to exclude from each border edge when spot finding
* wavelengths_nm - wavelengths of each color channel. This is used the calculate the PSF width
* tau - the frame period in seconds
* same_cutoff - distance cutoff in pixels of which spots are the same spot in different color channels
* contast_scale - multiplier for determining non-linearity of contrast sliders. Higher is  a wider range
* bleedthrough - fraction of donor intensity removed from acceptor intensity in the plots and exported traces. Only for 2 colors
* binning - factor of binning in the microscope image. Used to calculate the PSF width
* alignment_order - order of the polynomial used for alignment when loaded in alignment file
* ncpu - number of cpus to use in parallel when spotfinding in multiple frames
* maxiterations - maximum number of VBEM iterations to run when spotfinding
* magnification - Magnification of objective used in microscope
* threshold - relative threshold the lowerbound must decrease by during VBEM in spotfinding for the algorithm to converge
* ncolors - default number of colors. Won't change anything unless you change the number in the transform dock
* nstates - number of spot classes to use during spotfinding
* nintegrate - number of pixels (in one dimension) that are used for integrating the PSF to get the intensity values
* pixel_size - length of the (square) pixels on the camera in nanometers. Andor 888 is 13300, Andor 897 is 16000.
* numerical_aperture - the NA of the objective used to the collect the movie
