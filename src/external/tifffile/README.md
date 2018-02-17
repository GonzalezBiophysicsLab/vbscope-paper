# Readme

## To install
Run
```sh
python setup.py build_ext --inplace
```
## to use
Add the following to a script

```python
import sys
sys.path.append('/path/to/tifffile/directory')
import tifffile

fname = 'blah blah'

## this is a np.ndarray
d = tifffile.TiffFile(fname).asarray()
```
