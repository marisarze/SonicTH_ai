# Inspired by https://keon.io/deep-q-learning/
import retro
import random
import gym
import math
import numpy as np
import random
import time
import cv2
from matplotlib import pyplot as plt
from collections import deque
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import CuDNNLSTM
from keras.layers import Flatten
from keras.layers import RepeatVector
from keras import losses
from keras import regularizers

class SonicAgent():
    def __init__(self, n_episodes=2000, max_env_steps=None, state_len=3, alpha = 0.05, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_tau=3000, batchsize=1):
        self.state_len = state_len
        self.batchsize = batchsize
        self.state_memory = deque(maxlen=self.state_len)
        self.env = retro.make(game='SonicAndKnuckles3-Genesis', state='AngelIslandZone.Act1', scenario='contest')
        self.env.reset()
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_tau = epsilon_tau
        self.n_episodes = n_episodes
        self.chosen = 0
        self.xdata = deque(maxlen = self.state_len)
        self.ydata = deque(maxlen = self.state_len)
        self.docx = []
        self.docy = []
        #if max_env_steps is not None: self.env._max_episode_steps = max_env_steps

        # Init model
        
        input_img = Input(shape=(None, 144, 192, 3))
        x = TimeDistributed(Conv2D(150, (8, 8), activation='relu', padding='same'))(input_img)
        x = TimeDistributed(BatchNormalization())(x)
        print(x.shape)
        x = TimeDistributed(Conv2D(100, (8, 8), activation='relu', padding='same'))(x)
        x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)
        x = TimeDistributed(BatchNormalization())(x)
        print(x.shape)
        x = TimeDistributed(Conv2D(100, (6, 6), activation='relu', padding='same'))(x)
        x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)
        x = TimeDistributed(BatchNormalization())(x)
        print(x.shape)
        x = TimeDistributed(Conv2D(80, (6, 6), activation='relu', padding='same'))(x)
        x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)
        x = TimeDistributed(BatchNormalization())(x)
        print(x.shape)

        x = TimeDistributed(Conv2D(80, (9, 12), activation='relu', padding='same'))(x)
        encoded = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)
        #x = TimeDistributed(BatchNormalization())(x)
        #encoded = TimeDistributed(Conv2D(80, (8, 8), activation='relu', padding='same'))(x)
        self.encoder = Model(input_img, encoded)

        enc_shape = self.encoder.layers[-1].output_shape
        print(enc_shape[1:])
        input_encoded = Input(shape=enc_shape[1:])
        x = TimeDistributed(BatchNormalization())(input_encoded)

        x = TimeDistributed(Conv2D(80, (9, 12), activation='relu', padding='same'))(x)
        x = TimeDistributed(UpSampling2D((2, 2)))(x)
        x = TimeDistributed(BatchNormalization())(x)
        print(x.shape)

        x = TimeDistributed(Conv2D(100, (6, 6), activation='relu', padding='same'))(x)
        x = TimeDistributed(UpSampling2D((2, 2)))(x)
        x = TimeDistributed(BatchNormalization())(x)
        print(x.shape)

        x = TimeDistributed(Conv2D(100, (6, 6), activation='relu', padding='same'))(x)
        x = TimeDistributed(UpSampling2D((2, 2)))(x)
        x = TimeDistributed(BatchNormalization())(x)
        print(x.shape)

        x = TimeDistributed(Conv2D(150, (8, 8), activation='relu', padding='same'))(x)
        x = TimeDistributed(UpSampling2D((2, 2)))(x)
        x = TimeDistributed(BatchNormalization())(x)
        print(x.shape)

        decoded = TimeDistributed(Conv2D(3, (8, 8), activation='tanh', padding='same'))(x)
        print(decoded.shape)


        decoder = Model(input_encoded, decoded)
        self.decoder = Model(input_encoded, decoded)
        self.acoder = Model(input_img, self.decoder(self.encoder(input_img)))
        

        #sgd = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.acoder.compile(loss='mean_squared_error', optimizer=adam)
        self.encoder.compile(loss='mean_squared_error', optimizer=adam)
        self.decoder.compile(loss='mean_squared_error', optimizer=adam)
        self.acoder.summary()

    

    def convert_order(self, order):
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
        act = np.array([B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z])
        return act, adesc, adescn



    def choose_action(self, state_col, epsilon):
        order = random.randint(0,7)
        return self.convert_order(order)
        

    def get_epsilon(self, t):
        return self.epsilon_min + (1 - self.epsilon_min) * math.exp(-1 * t / self.epsilon_tau)

    def process_state_deque(self, state_deque):
        return np.reshape(np.true_divide(np.array(state_deque), 255.0) , (1, len(state_deque), 144, 192, 3)).astype(np.float32)

    def rgb2gray(self, imarray):
        resized = cv2.resize(imarray,(192,144))
        return np.reshape(np.array(resized, dtype=np.float32), (1, 144, 192, 3))
        #np.dot(imarray[...,:3], [0.299, 0.587, 0.114])

    def process_memory(self, state, adescn, reward, next_state, done):
        #prex = np.reshape(np.array(state, dtype=np.float32), (1, 140, 200, 3))
        #self.acoder.train_on_batch(prex,prex)
        self.xdata.append(self.rgb2gray(state))
        if len(self.xdata) == self.state_len:
            self.docx.append(self.process_state_deque(self.xdata))
        if (len(self.docx) == 1201) or done:
            self.replay()

    def replay(self):
        #self.xdata.clear()
        #self.ydata.clear()
        self.docx = np.reshape(np.array(self.docx, dtype=np.float32), (len(self.docx), self.state_len, 144, 192, 3))
        self.acoder.fit(self.docx, self.docx, batch_size = self.batchsize, verbose=1, epochs=1, shuffle=True)
        self.docx, self.docy = [], []

        

    def run(self):
        
        self.scores = deque(maxlen=100)
        cur_mem = deque(maxlen = self.state_len)
        self.mean = np.mean(self.scores)
        for e in range(self.n_episodes):
            state = self.env.reset()
            cur_mem.append(self.rgb2gray(state))
            action, adesc, adescn = self.choose_action(cur_mem, e)
            state1 = np.copy(state)
            adescn1 = adescn
            done = False
            i = 0
            episode_rew = 0
            tt = 0
            trew = 0
            time0 = time.time()
            self.steps = 0
            while True:
                if (tt == 10) or done:
                    cur_mem.append(self.rgb2gray(state))
                    action, adesc, adescn = self.choose_action(cur_mem, self.get_epsilon(e+1))
                    next_state, reward, done, info = self.env.step(action)
                    self.env.render()
                    state0 = np.copy(state1)
                    adescn0 = adescn1
                    state1 = np.copy(state)
                    adescn1 =adescn
                    self.process_memory(state0, adescn0, trew, state1, done)
                    trew = 0
                    tt = 0
                    self.steps += 1
                    if (self.steps == 500) and e>0:
                        self.predicted = self.acoder.predict(self.process_state_deque(cur_mem))
                        fig, axes = plt.subplots(1, self.state_len)
                        fig.set_size_inches(10, 10)
                        for im in range(self.state_len):
                            axes[im].imshow(self.predicted[-1,im,:])
                        plt.show(block=False)
                        plt.pause(3)
                        plt.close()
                        self.steps = 0
                        #resized = cv2.resize(state,(200,140))
                        
                        #axes[0].imshow(state)
                        #axes[1].imshow(resized)
                        #plt.show(block=False)
                        ##plt.pause(3)
                        #plt.gcf().canvas.flush_events()
                        #plt.show(block=False)
                        #self.steps = 0
                    if done:
                        print('epoch: ', e)
                        self.state_memory.clear()
                        print(time.time()-time0)
                        cur_mem.clear()
                        self.scores.append(episode_rew)
                        self.mean = np.mean(self.scores)
                        break
                else:
                    if adesc == 'B':
                        action, adesc, adescn = self.convert_order('pass')
                    else:
                        action, adesc, adescn = self.convert_order(adesc)
                    next_state, reward, done, info = self.env.step(action)
                    #self.env.render()
                state = np.copy(next_state)
                tt += 1
                i += 1
                trew += reward
                episode_rew += reward
                
                if self.chosen == 1:
                    self.chosen = 0
                    print('episode: , {}, episode_reward: , {:.6}, action: , {}, mean: {:.5}'.format(e,episode_rew, adesc, self.mean))



if __name__ == '__main__':
    agent = SonicAgent()
    agent.run()







