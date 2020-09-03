from matplotlib import pyplot as plt
import numpy as np

r = 2
x0 = 0.9
dx = -0.1
aspace = 5
d = 0.5
A = 1
beta = 1/2/dx
C = abs(beta * A)
Z = 0.99
start = np.clip(x0-dx-0.01, 0,1)
finish = np.clip(x0+dx+0.01, 0,1)
probs = np.linspace(start, finish, 10000, dtype=np.float64)
distr = np.zeros((aspace,), dtype=np.float64)
L = []
C1 = []
C2 = []
ratios = []
entropies = []
for x in probs:
    distr[0] = x
    distr[1:] = (1-x)/(aspace-1)
    entropy = -np.sum(distr * np.log(distr+1e-3), axis=-1) / np.log(aspace)
    entropies.append(entropy.astype(np.float64))
    x1 = np.clip(x-dx, 0,1)
    x2 = np.clip(x+dx, 0,1)
    addit = C * (x-x0) ** r
    base = - A * x
    ratios.append(addit/base)
    L.append(addit+base)
crit = 1/2/beta

print(probs[np.argmin(L)])
print(probs[np.argmax(L)])
dl = []
for i in range(len(L)-1):
    dl.append((L[i+1]-L[i])/(probs[i+1]-probs[i]))
dl = np.array(dl)

entropies = np.array(entropies)
L = np.array(L)
plt.plot(probs, ratios)
#plt.plot(probs[:-1],dl)
#plt.plot(probs,entropies)
plt.ylabel('loss')
# plt.ylabel('loss')
plt.show()

