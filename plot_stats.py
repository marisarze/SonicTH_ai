import json
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open(r'D:\sonic_models\result.json', 'r') as file:
    stat = json.load(file)

x = np.array(stat['x'])
y = -np.array(stat['y'])
s = np.array(stat['some_map'])
e = np.array(stat['entropy_map']) 
r = np.array(stat['ireward_map'])

d = 8

#r = np.clip(r, 0,100)

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
#aa = a[::d]

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(xx,yy,ss)
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('ratio')
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xx,yy,ee)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('entropy')
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(xx,yy,aa)
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('artificial')
# plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(xx,yy,rr)
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('rewards')
# plt.show()

total = []
ind = 0
sum = 0
for elem in stat['cutted']:
    elem = int(elem)
    sum = 0
    count = 0
    for i in r[ind:ind+elem]:
        sum += i * (0.999 ** count)
        total.append(sum)
        count += 1
    ind += elem
total = np.array(total)
tt = total[::d]

print(total.shape, x.shape, y.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xx,yy,tt)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('total')
plt.show()


score = stat['entropy']
episodes = stat['episodes_numbers']
steps_list = np.array(stat['steps_list']) / 1e6
steps = []
sum = 0
for elem in steps_list:
    sum += elem
    steps.append(sum)
    
print(len(score))
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(steps, score)
ax.set_xlabel('agent steps (millions)')
ax.set_ylabel('entropy')
plt.show()

rscore = stat['mean100_external_rewards']
episodes = stat['episodes_numbers']
print(len(score))
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(steps, rscore)
ax.set_xlabel('agent steps (millions)')
ax.set_ylabel('ereward_mean')
#ax.set_xscale('log')
plt.show()

stds = stat['action_std']
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(steps, stds)
ax.set_xlabel('some index')
ax.set_ylabel('action_stds')
plt.show()


fig = plt.figure()
ax = fig.add_subplot()
ax.plot(steps_list)
ax.set_xlabel('some index')
ax.set_ylabel('episode length')
plt.show()
