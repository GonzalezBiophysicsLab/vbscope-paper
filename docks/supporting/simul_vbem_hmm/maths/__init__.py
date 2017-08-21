from kullback_leibler_divergence import dkl_normalgamma, dkl_dirichlet, dkl_tmatrix
from posterior_updates import update_tmatrix, update_rho, update_normals
from expectations import vbem_1d_ln_p_x_z as E_ln_p_x_z
from expectations import vbem_1d_ln_A as E_ln_A
from expectations import vbem_1d_ln_pi as E_ln_pi
from math_fxns import psi

from normal_numba import p_normal
