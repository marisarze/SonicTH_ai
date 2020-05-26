from matplotlib import pyplot as plt
import numpy as np
import math
N = 1000
r = 2.0

start = 0
finish = 1
x = np.linspace(start, finish, N)
y = np.zeros((N,))
d = 0.5


def wfunc(x, d, r):
    z = 1/(1+math.exp(-(x-d)))-0.5
    return -(abs(z) ** (1/r)) if z<0 else z ** (1/r)

def normed_wfunc(x, d, r):
    z1= (wfunc(x,d,r)-wfunc(0,d,r))/(wfunc(1,d,r)-wfunc(0,d,r))
    return z1


for i in range(N):
    y[i] = normed_wfunc(x[i], d, r)

# plt.plot(probs,L)
plt.plot(x,y)
# plt.xlabel('entropy')
# plt.ylabel('loss')
plt.show()

fig = plt.figure()
ax = fig.add_subplot()
ss =  2*x - np.array(list(map(lambda x: math.exp(-x), x)))
ax.plot(x, x ** 0.1)
plt.show()