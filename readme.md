# Installation
Should work on best on Unix-based systems (Mac and Linux); Mostly works on Windows

## Quick Start
1. Install miniconda (64 bits)
2. Install python libraries
2. Install git
3. git clone this repository
4. Make shortcuts
5. Launch smfret_plotter

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
alias smfret_plotter="python /home/username/directory/to/vbscope/smfret_plotter.py"
```

## Mac hints
1. Install miniconda (download link: https://conda.io/miniconda.html, terminal >> sh ~/Downloads/name_of_the_miniconda_file.sh, close terminal, open terminal)
2. Install git (terminal >> xcode-select --install)
3. python libraries (terminal >> see above)
4. shortcuts (text edit >> ~/.bash_profile)
5. Launch (terminal >> vbscope)

## Hints
1. You must upload an ssh key to gitlab in order to download/upload using git. Gitlab >> Settings >> SSH keys. Follow link out for system specific help doing this.
