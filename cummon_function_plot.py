from matplotlib import pyplot as plt
import numpy as np
import math
N = 1000
r = 1


start = 0
finish = 1
x = np.linspace(start, finish, N)
y = np.zeros((N,))
d = 0.5
nd = 1- d
speed = 10

def wfunc(x, d, r):
    z = 1/(1+math.exp(-speed*(x-d)))-0.5
    return -(abs(z) ** (1/r)) if z<0 else z ** (1/r)

def normed_wfunc(x, d, r):
    z1= (wfunc(x,d,r)-wfunc(0,d,r))/(wfunc(1,d,r)-wfunc(0,d,r))
    return z1

pw = 1
for i in range(N):
    y[i] = normed_wfunc(x[i], d, r)
df = np.zeros((N-1,))
dy = np.zeros((N-1,))
for i in range(N-1):
    dy[i] = (y[i+1]-y[i])/(x[i+1]-x[i])
    df[i] = (x[i+1] ** pw-x[i] ** pw)/(x[i+1]-x[i])

# plt.plot(probs,L)
plt.plot(x,y)
plt.plot(x, 0.2* x ** 1.4 )
# plt.xlabel('entropy')
# plt.ylabel('loss')
plt.show()

plt.plot(x[:-1],dy)
plt.plot(x[:-1],df)
# plt.xlabel('entropy')
# plt.ylabel('loss')
plt.show()