import matplotlib.pyplot as plt
import numpy as np
import normal_dist as nd
import poisson_dist as pd

n = 9
mu = 1000.
std = 50.#mu**2.
m = 10000
d = np.random.randn(n,m)*std + mu

d = np.random.poisson(lam=1000.,size=(n,m))-1000.


# plt.figure()
# plt.hist(d.max(0),bins=30,alpha=.5,normed=1,histtype='stepfilled')
# plt.hist(d.mean(0),bins=30,alpha=.5,normed=1,histtype='stepfilled')
# plt.hist(d.min(0),bins=30,alpha=.5,normed=1,histtype='stepfilled')
#
#
# mu,var = nd.fit_maxval_normal(d.max(0),n)
#
# x = np.linspace(d.min(),d.max(),1000)
# plt.plot(x,nd.p_normal(x,mu,var),'k')
# plt.hist(d.flatten(),bins=30,alpha=.5,normed=1,histtype='stepfilled',zorder=-2)
# plt.plot(x,nd.maxval_normal(x,n,mu,var),'k')
# plt.plot(x,nd.minval_normal(x,n,mu,var),'k')
# plt.plot(x,nd.p_normal(x,mu,var/n),'k')
# plt.show()
#
#
#
plt.figure()
plt.hist(d.max(0),bins=30,alpha=.5,normed=1,histtype='stepfilled')
plt.hist(d.mean(0),bins=30,alpha=.5,normed=1,histtype='stepfilled')
plt.hist(d.min(0),bins=30,alpha=.5,normed=1,histtype='stepfilled')


print np.var(d.min(0)),d.min(0).mean()
print np.var(d.min(0))-d.min(0).mean()
print np.var(d)

x = np.linspace(d.min(),d.max(),1000)

# bg = 1000.
# plt.plot(x,np.exp(pd.ln_maxval_poisson(x+bg,9,1000.)),'k')
# plt.plot(x,np.exp(pd.ln_minval_poisson(x+bg,9,1000.)),'k')
#
lam = pd.fit_minval_poisson(d.min(0),n)
# print lam
plt.plot(x,pd.p_poisson(x+lam[1],lam[0]),'k')
plt.hist(d.flatten(),bins=50,alpha=.5,normed=1,histtype='stepfilled',zorder=-2)
plt.plot(x,np.exp(pd.ln_maxval_poisson(x+lam[1],n,lam[0])),'k')
plt.plot(x,np.exp(pd.ln_minval_poisson(x+lam[1],n,lam[0])),'k')
plt.plot(x,pd.p_poisson(x+lam[1],lam[0]),'k')
plt.show()