import matplotlib.pyplot as plt
from simulate_data import simulate
import numpy as np

class plott():
	def __init__(self,t,d,q,result):
		self.t = t
		self.d = d
		self.q = q
		self.i = 0
		self.result = result
		self.f,self.a = plt.subplots(1)


		cm = plt.cm.jet
		# for j in range(self.result.nstates):
		# 	m = self.result.m[j]
		# 	s = np.sqrt(self.result.b[j]/self.result.a[j])
		# 	c = cm(float(j)/self.result.nstates)
		# 	a = self.result.pistar[j]
		# 	self.a.axhspan(m-3.*s,m+3.*s,color=c,alpha=.3,zorder=-3)
		self.ll = self.a.plot(self.t[self.i],self.d[self.i],'o',alpha=.3,color='k')[0]

		self.l = []
		for j in range(self.result.nstates):
			c = cm(float(j)/self.result.nstates)
			# cut = self.result.ln_p_x_z[self.i].argmax(1) == j
			cut = self.result.viterbi[self.i] == j
			self.l.append(self.a.plot(self.t[self.i][cut],self.d[self.i][cut],'o',alpha=.3,color=c)[0])
		# self.l = self.a.plot(self.t[self.i],self.d[self.i],color='b',lw=1.,alpha=.8)[0]
		self.v = self.a.plot(self.t[self.i],np.zeros_like(self.d[self.i]),color='k',lw=1.,alpha=.5)[0]
		self.vv = self.a.plot(self.t[self.i],np.zeros_like(self.d[self.i]),color='r',lw=1.,alpha=.5)[0]

		self.a.set_xlim(0,self.t[self.i][-1])
		self.a.set_ylim(-.1,1.1)
		self.f.canvas.mpl_connect('key_press_event',self.keypush)
		plt.show()
		self.update


	def keypush(self,event):
		if event.key == 'right':
			self.i += 1
		elif event.key == 'left':
			self.i -= 1
		if self.i < 0:
			self.i = 0
		if self.i >= len(self.d):
			self.i = len(self.d) - 1
		self.update()

	def update(self):
		self.ll.set_data(self.t[self.i],self.d[self.i])
		for j in range(self.result.nstates):
			# cut = self.result.ln_p_x_z[self.i].argmax(1) == j
			cut = self.result.viterbi[self.i] == j
			self.l[j].set_data(self.t[self.i][cut],self.d[self.i][cut])

		self.vv.set_data(self.t[self.i],self.result.m[self.result.viterbi[self.i]])
		self.v.set_data(self.t[self.i],self.q[self.i])

		self.a.set_xlim(self.t[self.i][0],self.t[self.i][-1])
		self.a.set_ylim(-.1,1.1)
		self.a.set_title(str(self.i))
		self.f.canvas.draw()
		plt.draw()


#
# n = 100
# tpb = 4000
# l = np.random.exponential(tpb,size=n)
# l[l<5.] = 5.
# t = []
# d = []
# for i in range(n):
# 	trace = simulate(int(l[i]))
# 	t.append(trace.x)
# 	d.append(trace.y)
# 	np.random.seed()
#
# p = plot(t,d)
