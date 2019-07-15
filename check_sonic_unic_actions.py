
import retro
import numpy as np
from collections import deque
import cv2
from matplotlib import pyplot as plt
import random
env = retro.make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1', scenario='contest')
episodes = 1

def convert_order(order):
    B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    if (order == 'pass') or (order == 0):
        adesc = 'pass'
        adescn = 0
    elif (order == 'L') or (order == 1):
        LEFT = 1
        adesc = 'L'
        adescn = 1
    elif (order == 'R') or (order == 2):
        RIGHT = 1
        adesc = 'R'
        adescn = 2
    elif (order == 'LD') or (order == 3):
        LEFT = 1
        DOWN = 1
        adesc = 'LD'
        adescn = 3
    elif (order == 'RD') or (order == 4):
        RIGHT = 1
        DOWN = 1
        adesc = 'RD'
        adescn = 4
    elif (order == 'D') or (order == 5):
        DOWN = 1
        adesc = 'D'
        adescn = 5
    elif (order == 'DB') or (order == 6):
        DOWN = 1
        B = 1
        adesc = 'DB'
        adescn = 6
    elif (order == 'B') or (order == 7):
        B = 1
        adesc = 'B'
        adescn = 7
    elif (order == 'A') or (order == 7):
        B = 1
        adesc = 'A'
        adescn = 8
    act = np.array([B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z])
    return act


m = deque(maxlen=1)



for e in range(episodes):
    obs = env.reset()
    aord = 'B'
    
    done = False
    t = 0
    ss = 0
    while not done:
        env.render()
        action = convert_order(random.randint(0,8))
        obs, rew, done, info = env.step(action)
        resized = cv2.resize(obs,(128, 90))
        if ss>1000:
            fig, axes = plt.subplots(1, 1)
            fig.set_size_inches(10, 10)
            axes.imshow(resized)
            plt.show(block=False)
            plt.pause(1)
            plt.close()
            ss = 0
        t += 1
        ss += 1
