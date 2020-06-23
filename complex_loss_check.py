from matplotlib import pyplot as plt
import numpy as np

r = 4
aspace = 5
d = 0.5
A = 1
C = abs(0.1* A)
Z = 0.8
start = 0.9
finish = 1
probs = np.linspace(start, finish, 10000, dtype=np.float64)
distr = np.zeros((aspace,), dtype=np.float64)
L = []
entropies = []
for x in probs:
    distr[0] = x
    distr[1:] = (1-x)/(aspace-1)
    entropy = -np.sum(distr * np.log(distr), axis=-1) / np.log(aspace)
    entropies.append(entropy.astype(np.float64))
    L.append(- A * x / Z + C * (entropy-1) ** r)
print(-A/Z)
dl = []
for i in range(len(L)-1):
    dl.append((L[i+1]-L[i])/(probs[i+1]-probs[i]))
dl = np.array(dl)

entropies = np.array(entropies)
L = np.array(L)
plt.plot(probs, L)
#plt.plot(probs[:-1],dl)
#plt.plot(probs,entropies)
plt.ylabel('entropy')
# plt.ylabel('loss')
plt.show()

