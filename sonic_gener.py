# Inspired by https://keon.io/deep-q-learning/
import retro
import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import CuDNNLSTM
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras import losses
from keras.models import clone_model

class SonicAgent():
    def __init__(self, n_steps=1000, memlen=50, max_env_steps=None, gamma=1.0, epsilon=1.0, epsilon_min=0.01, epsilon_tau=10, alpha=0.01, alpha_decay=0.01, quiet=False):
        self.memory = deque(maxlen=memlen)
        self.memlen = memlen
        self.env = retro.make(game='SonicAndKnuckles3-Genesis', state='AngelIslandZone.Act1')
        self.env.reset()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_tau = epsilon_tau
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_steps = n_steps
        #if max_env_steps is not None: self.env._max_episode_steps = max_env_steps

        # Init model
        
        model = Sequential()
        model.add(TimeDistributed(Conv2D(16, kernel_size=4, activation='relu'), input_shape=(None, 224, 320, 3)))
        model.add(TimeDistributed(Conv2D(16, kernel_size=4, activation='relu')))
        model.add(TimeDistributed(Flatten()))
        model.add(CuDNNLSTM(10, return_sequences=True))
        #model.add(TimeDistributed(Dropout(0.5)))
        model.add(CuDNNLSTM(10, return_sequences=True))
        #model.add(TimeDistributed(Dropout(0.5)))
        model.add(TimeDistributed(Dense(8, activation='relu')))
        #adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss='mean_squared_error', optimizer='sgd')
        model.summary()
        self.model = model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        st = np.reshape(np.array(state), (1, 1, 224, 320, 3))
        act_sp = self.model_copy.predict(st)[0][-1]
        bB, bA, bMODE, bSTART, bUP, bDOWN, bLEFT, bRIGHT, bC, bY, bX, bZ = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        temp = [bB, bA, bMODE, bSTART, bUP, bDOWN, bLEFT, bRIGHT, bC, bY, bX, bZ]
        if np.argmax(act_sp) == 1:
            bLEFT = 1
        if np.argmax(act_sp) == 2:
            bRIGHT = 1
        if np.argmax(act_sp) == 3:
            bLEFT = 1
            bDOWN = 1
        if np.argmax(act_sp) == 4:
            bRIGHT = 1
            bDOWN = 1
        if np.argmax(act_sp) == 5:
            bDOWN = 1
        if np.argmax(act_sp) == 6:
            bDOWN = 1
            bB = 1
        if np.argmax(act_sp) == 7:
            bB = 1

        return self.env.action_space.sample() if (np.random.random() <= epsilon) else temp

    def get_epsilon(self, t):
        return self.epsilon_min + (1 - self.epsilon_min) * math.exp(-1 * t / self.epsilon_tau)

    def replay(self):
        x_batch, y_batch = [], []
        for state, action, reward, next_state, done in self.memory:
            st = np.reshape(np.array(state), (1, 1, 224, 320, 3))
            next_st = np.reshape(np.array(next_state), (1, 1, 224, 320, 3))
            y_target = self.model_copy.predict(st)
            y_target[0][-1][action] = reward if done else reward + self.gamma * np.max(self.model_copy.predict(next_st)[0][-1])
            x_batch.append(state)
            y_batch.append(y_target[0][0])
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        x_batch = np.reshape(x_batch, (1, len(self.memory), 224, 320, 3))
        y_batch = np.reshape(y_batch, (1, len(self.memory), 8))
        return x_batch, y_batch

    def run(self):
        self.model.fit_generator(self.game_gener(),steps_per_epoch=1000000, epochs=1, 
                 verbose=1)
        

    def game_gener(self):
        scores = deque(maxlen=100)
        i = 0
        e = 0
        while i < self.n_steps:
            state = self.env.reset()
            done = False
            e += 1 
            while not done:
                self.model_copy = clone_model(self.model)
                self.model_copy.set_weights(self.model.get_weights())
                action = self.choose_action(state, self.get_epsilon(e+1))
                next_state, reward, done, info = self.env.step(action)
                print(info)
                self.env.render()
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                yield self.replay()
            



if __name__ == '__main__':
    agent = SonicAgent()
    agent.run()






#import retro


#print(retro.data.list_games())
#print(retro.data.list_states('SonicAndKnuckles3-Genesis'))
#def main():
#    env = retro.make(game='SonicAndKnuckles3-Genesis', state='AngelIslandZone.Act1')
#    obs = env.reset()
#    obs, rew, done, info = env.step(env.action_space.sample())
#    N = 1
#    print(obs.shape)
#    print(env.action_space.shape)
#    totalrew = 0
#    while N>0:
#        obs, rew, done, info = env.step(env.action_space.sample())
#        totalrew += rew
#        print('++', info, '++')
#        env.render()
#        if done:
#            obs = env.reset()
#            N -= 1
#            print('-------------game is over---------------------------')


#if __name__ == '__main__':
#    main()
