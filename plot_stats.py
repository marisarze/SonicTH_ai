import json
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open(r'D:\sonic_models\results.json', 'r') as file:
    stat = json.load(file)


def wfunc(x, d, r, speed):
            z = 1/(1+math.exp(-speed*(x-d)))-0.5
            return -(abs(z) ** (1/r)) if z<0 else z ** (1/r)

def normed_wfunc(x, d, r, speed):
    z1= (wfunc(x, d, r, speed)-wfunc(0, d, r, speed))/(wfunc(1, d, r, speed)-wfunc(0, d, r, speed))
    return z1
d = 8

x = np.array(stat['x'])
y = -np.array(stat['y'])
s = np.array(stat['some_map'])
e = np.array(stat['entropy_map']) 
r = np.array(stat['ireward_map'])
#a = -100 *(1-e) + s

weights = []
for elem in e:
    weights.append(normed_wfunc(elem, 0.9, 1.0, 50.0))
weights = np.array(weights)
ind = 2
weights_2 = np.zeros((len(weights)))
for i in range(ind, len(weights)-ind):
    weights_2[i] = np.mean(weights[i-ind:i+ind+1]) ** (1/(2*ind+1))
weights_2[:ind] = weights_2[ind]
weights_2[-ind:] = weights_2[-ind]



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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(xx,yy,ss)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('ratio')
plt.show()

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
    for i in r[ind:ind+elem]:
        sum += i
        total.append(sum)
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


score = stat['mean100_entropy']
episodes = stat['episodes_numbers']
steps = np.array(stat['steps_list']) / 1e6
print(len(score))
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(steps, score)
ax.set_xlabel('agent steps (millions)')
ax.set_ylabel('entropy')
plt.show()

stds = stat['action_std']
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(steps, stds)
ax.set_xlabel('some index')
ax.set_ylabel('action_stds')
plt.show()

steps_list = np.array(stat['steps_list'])
new_list =[]
for i in range(len(steps_list)):
    new_list.append(np.mean(steps_list[i]-steps_list[i-50:i])/50)

new_list = np.array(new_list)

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(new_list)
ax.set_xlabel('some index')
ax.set_ylabel('episode length')
plt.show()