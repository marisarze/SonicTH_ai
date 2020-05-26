import json
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


with open(r'D:\sonic_models\results.json', 'r') as file:
    stat = json.load(file)




x = np.array(stat['x'])
y = -np.array(stat['y'])
s = np.array(stat['some_map'])
s -= np.mean(s)
e = np.array(stat['entropy_map']) 
r = np.array(stat['ireward_map'])
a = r * e ** 4
d = 12
# randomize = np.arange(steps)
# np.random.shuffle(randomize)
# x = x[randomize]
# y = y[randomize]
# s = s[randomize]
ee = e[::d]
xx = x[::d]
yy = y[::d]
ss = s[::d]
rr = r[::d]
aa = a[::d]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xx,yy,ss)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('ratio')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xx,yy,ee ** 4)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('entropy')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xx,yy,aa)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('artificial')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xx,yy,rr)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('rewards')
plt.show()


score = stat['mean100_entropy']
episodes = stat['episodes_numbers']
steps = stat['steps_list']
print(len(score))
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(steps, score)
plt.show()
