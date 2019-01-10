# Inspired by https://keon.io/deep-q-learning/
import retro
import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import CuDNNLSTM
from keras.layers import Conv2D
from keras.layers import Flatten
from keras import losses

class SonicAgent():
    def __init__(self, n_episodes=1000, memlen=5, max_env_steps=None, gamma=1.0, epsilon=1.0, epsilon_min=0.01, epsilon_tau=100, alpha=0.01, alpha_decay=0.01, batch_size=64, quiet=False):
        self.memory = deque(maxlen=memlen)
        self.memlen = memlen
        self.env = retro.make(game='SonicAndKnuckles3-Genesis', state='AngelIslandZone.Act1', scenario='contest')
        self.env.reset()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_tau = epsilon_tau
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        #if max_env_steps is not None: self.env._max_episode_steps = max_env_steps

        # Init model
        
        model = Sequential()
        model.add(TimeDistributed(Conv2D(48, kernel_size=16, activation='relu'), input_shape=(None, 224, 320, 3)))
        model.add(TimeDistributed(Conv2D(24, kernel_size=8, activation='relu')))
        model.add(TimeDistributed(Conv2D(16, kernel_size=4, activation='relu')))
        model.add(TimeDistributed(Flatten()))
        model.add(CuDNNLSTM(10, return_sequences=True))
        #model.add(TimeDistributed(Dropout(0.1)))
        model.add(CuDNNLSTM(10, return_sequences=True))
        #model.add(TimeDistributed(Dropout(0.1)))
        model.add(TimeDistributed(Dense(8, activation='relu')))
        #adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss='mean_squared_error', optimizer='sgd')
        model.summary()
        self.model = model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        if (np.random.random() <= epsilon):
            return self.env.action_space.sample()
        else:
            st = np.reshape(np.array(state), (1, 1, 224, 320, 3))
            act_sp = self.model.predict(st)[0][-1]
            #print(act_sp)
            bB, bA, bMODE, bSTART, bUP, bDOWN, bLEFT, bRIGHT, bC, bY, bX, bZ = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            argm = np.argmax(act_sp)
            
            if argm == 0:
                pass
            elif argm == 1:
                bLEFT = 1
            elif argm == 2:
                bRIGHT = 1
            elif argm == 3:
                bLEFT = 1
                bDOWN = 1
            elif argm == 4:
                bRIGHT = 1
                bDOWN = 1
            elif argm == 5:
                bDOWN = 1
            elif argm == 6:
                bDOWN = 1
                bB = 1
            else:
                bB = 1
            temp = np.array([bB, bA, bMODE, bSTART, bUP, bDOWN, bLEFT, bRIGHT, bC, bY, bX, bZ])
            print('action: ', temp)
            return temp
        

    def get_epsilon(self, t):
        return self.epsilon_min + (1 - self.epsilon_min) * math.exp(-1 * t / self.epsilon_tau)

    def replay(self):
        x_batch, y_batch = [], []
        for state, action, reward, next_state, done in self.memory:
            st = np.reshape(np.array(state), (1, 1, 224, 320, 3))
            next_st = np.reshape(np.array(next_state), (1, 1, 224, 320, 3))
            y_target = self.model.predict(st)
            y_target[0][-1][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_st)[0][-1])
            x_batch.append(state)
            y_batch.append(y_target[0][0])
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        x_batch = np.reshape(x_batch, (1, len(self.memory), 224, 320, 3))
        y_batch = np.reshape(y_batch, (1, len(self.memory), 8))
        self.model.fit(x_batch, y_batch, batch_size = 1, verbose=0)


    def run(self):
        scores = deque(maxlen=100)
        e = 0
        while e <= self.n_episodes:
            e += 1
            state = self.env.reset()
            done = False
            i = 0
            episode_rew = 0
            tt = 0
            ep_maxx = 5002
            while not done:
                action = self.choose_action(state, self.get_epsilon(e))
                next_state, reward, done, info = self.env.step(action)
                print('episode: ',e,'  reward: ',reward)
                tt += 1
                self.env.render()
                self.remember(state, action, reward, next_state, done)
                if (tt > 30):
                    self.replay()
                    tt = 0
                state = next_state
                i += 1
                episode_rew += reward
    
            scores.append(episode_rew)
            mean_score = np.mean(scores)
            print('episode: ',e, 'mean_score: ', mean_score)


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