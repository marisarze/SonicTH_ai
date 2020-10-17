from matplotlib import pyplot as plt
import numpy as np

r = 2
x0 = 1
dx = -0.1
aspace = 5
d = 0.5
A = -1
beta = 1/2/dx
C = 0.0 * abs(beta * A)
C2 = 0.07 * abs(beta * A)
Z = 0.99
start = 0.0 #np.clip(x0-dx-0.01, 0,1)
finish = 1.0 #np.clip(x0+dx+0.01, 0,1)
probs = np.linspace(start, finish, 10000, dtype=np.float32)
distr = np.zeros((aspace,), dtype=np.float32)
L = []
ratios = []
entropies = []
for x in probs:
    distr[0] = x
    distr[1:] = (1-x)/(aspace-1)
    entropy = -np.sum(distr * np.log(distr+1e-3), axis=-1) / np.log(aspace)
    entropies.append(entropy.astype(np.float32))
    x1 = np.clip(x-dx, 0,1)
    x2 = np.clip(x+dx, 0,1)
    addit = C * (x-x0) ** r
    base = - A * x
    addit2 = C2 * x ** 4
    ratios.append(addit/base)
    L.append(addit2+base+addit)
crit = 1/2/beta

print(probs[np.argmin(L)])
print(probs[np.argmax(L)])
dl = []
for i in range(len(L)-1):
    dl.append((L[i+1]-L[i])/(probs[i+1]-probs[i]))
dl = np.array(dl)

entropies = np.array(entropies)
L = np.array(L)
plt.plot(probs, L)
#plt.plot(probs[:-1],dl)
#plt.plot(probs,entropies)
plt.ylabel('loss')
# plt.ylabel('loss')
plt.show()

