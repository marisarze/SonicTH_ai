import json
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


with open(r'D:\sonic_models\results.json', 'r') as file:
    stat = json.load(file)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.array(stat['x'])
y = -np.array(stat['y'])
s = np.array(stat['some_map'])

steps = len(x)
i = math.ceil(steps/10)
# randomize = np.arange(steps)
# np.random.shuffle(randomize)
# x = x[randomize]
# y = y[randomize]
# s = s[randomize]

xx = x[:i]
yy = y[:i]
ss = s[:i]
ax.scatter(xx,yy,ss)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('ratio')
plt.show()


score = stat['entropy']

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(score)
plt.show()