**Note**: we are in the process of developing a new package to perform spot-fiding in single-molecule TIRF microscopy data. This code is provided as legacy information.

# Installation
Should work on best on Unix-based systems (Mac and Linux); Mostly works on Windows

## Quick Start
1. Install miniconda (64 bits)
2. Install python libraries
2. Install git
3. git clone this repository
4. Make shortcuts
5. Launch vbscope

## Required python libraries
* pyqt 5.9.4
* numpy 1.11.3
* matplotlib 2.2.2
* scipy 0.19.0
* scikit-image 0.13.1
* numba 0.35
* h5py 2.7.1

### Install using conda
``` bash
conda install scikit-image scipy numpy matplotlib pyqt numba h5py 
```

## Make a Terminal Shortcut
To your rc file (eg ~/.bashrc, ~/.zshrc, ~/.profile, etc) add
``` bash
alias vbscope="python /home/username/directory/to/vbscope/vbscope.py"
```

## Mac hints
1. Install miniconda (download link: https://conda.io/miniconda.html, terminal >> sh ~/Downloads/name_of_the_miniconda_file.sh, close terminal, open terminal)
2. Install git (terminal >> xcode-select --install)
3. python libraries (terminal >> see above)
4. shortcuts (text edit >> ~/.bash_profile)
5. Launch (terminal >> vbscope)

## Hints
1. You must upload an ssh key to gitlab in order to download/upload using git. Gitlab >> Settings >> SSH keys. Follow link out for system specific help doing this.

# Parameters
Explanations of the parameters in the preferences:

## Background
* background_pixel_dist - for testing
* background_smooth_dist - for testing
* background_time_dist - for testing

## Channels
* channels_colors - the color used to plot each channel (string colors accepted)
* channels_wavelength - wavelengths of each color channel. This is used the calculate the PSF width

## Computer
* computer_ncpu - number of cpus to use in multiprocessing algorithms

## Intensity Extraction
* extraction_binning - factor of binning in the microscope image. Used to calculate the PSF width
* extract_magnification - magnification of objective used in microscope
* extract_ml_psf_maxiters - number of ML iterations
* extract_numerical_aperture - the NA of the objective used to the collect the movie
* extract_nintegrate - number of pixels (in one dimension) that are used for integrating the PSF to get the intensity values
* extract_pixel_size - length of the (square) pixels on the camera in nanometers. Andor 888 is 13300, Andor 897 is 16000
* extract_same_cutoff - distance cutoff in pixels of which spots are the same spot in different color channels

## Movie
* movie_playback_fps - upper-limit of frames per second for playing movie
* movie_tau - the frame period in seconds

## Plot
* plot_colormap - colormap of image
* plot_contrast_scale - non-linear contrast scaling for contrast dock
* plot_fontsize - fontsize in plots

## Render
* render_artist - metadata name
* render_codec - codec for movie rendering
* render_fps - frames per second of rendered movie
* render_renderer - program to use to render movie frames together
* render_title - metadata movie title

## Spot Finding
* spotfind_clip_border - number of pixels to exclude from each border edge when spot finding
* spotfind_maxiterations - maximum number of VBEM iterations to run when spotfinding
* spotfind_nsearch - number of pixels (in one dimension) that are used for finding a local min or local max. Can only be odd numbers
* spotfind_nstates - number of spot classes to use during spotfinding
* spot_threshold - relative threshold the lowerbound must decrease by during VBEM in spotfinding for the algorithm to converge

## Alignment
* Transform_alignment_order - order of the polynomial used for alignment when loaded in alignment file

## User Interface
* ui_bgcolor - background color
* ui_fontcolor - color of font
* ui_fontsize - font size
* ui_height - size of window
* ui_version - version number
* ui_width - size of window
