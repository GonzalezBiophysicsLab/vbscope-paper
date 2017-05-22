import numpy as np
from skimage.transform import AffineTransform


def icp(reference,transformee):
    '''
    both inputs are Nx2 vectors w/ coordinates of points. can be different numbers of points

    * reference is the set which everything will be transformed into
    * transformee is the set that will be transformed

    returns: the scikit-image affine transform function. To use this, input
    coordinates to be transformed (i.e., transformee) in Nx2 form... and it will return the transformed coordinates
    '''
    maxiters = 100

    c1 = reference.T
    c2 = transformee.T

    t = AffineTransform()
    last = np.inf
    for i in range(maxiters):
        ct2 = t(c2.T).T
        dr = np.sqrt((ct2[0,None,:] - c1[0,:,None])**2. + (ct2[1,None,:] - c1[1,:,None])**2.)
        rmin = dr.argmin(1)
        dst = c2[:,rmin]
        t.estimate(dst.T,c1.T)
        l = np.median(t.residuals(dst.T,c1.T))
        if np.isclose(l,last):
            return t
        else:
            last = l
    print "Didn't converge... which is weird."
    return t


### Example:
# from icp import icp
#
# transform = icp(c1.T,c2.T)
#
# plt.figure()
# plt.plot(c1[0],c1[1],'o')
# plt.plot(c2[0],c2[1],'o')
# ct2 = transform(c2.T).T
# plt.plot(ct2[0],ct2[1],'o')
# plt.show()
