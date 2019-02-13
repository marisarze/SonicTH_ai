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
from keras.optimizers import Adam
from keras.optimizers import SGD
from collections import deque
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, Dense, UpSampling2D
from keras.models import clone_model
from keras.layers import BatchNormalization
from keras.layers import TimeDistributed
from keras.layers import CuDNNLSTM
from keras.layers import Conv2D
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import concatenate
from keras import losses
from keras import regularizers
from keras.models import Model

class SonicAgent():
    def __init__(self, n_episodes=2000, max_env_steps=None, state_len=4, alpha = 0.1, gamma=0.999, epsilon=1.0, epsilon_min=0.05, epsilon_tau=500000, batchsize=1):
        self.state_len = state_len
        self.epsilon_max = 1
        self.batchsize = batchsize
        
        self.env = retro.make(game='SonicAndKnuckles3-Genesis', state='AngelIslandZone.Act1', scenario='contest')
        self.env.reset()
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_tau = epsilon_tau
        self.n_episodes = n_episodes
        self.chosen = 0
        self.state_small = deque(maxlen=self.state_len)
        
        self.xdata = deque(maxlen = self.state_len)
        self.ydata = deque(maxlen = self.state_len)
        self.sdata = deque(maxlen = self.state_len)
        self.docx = []
        self.docy = []
        self.docs = []
        self.encoder, self.decoder, self.acoder = self.build_coders()
        self.enc_shape = self.encoder.layers[-1].output_shape
        print(self.enc_shape)
        self.Amodel = self.build_model()
        self.Bmodel = self.build_model()
        self.adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.Amodel.compile(loss='mean_squared_error', optimizer=self.adam)
        self.Bmodel.compile(loss='mean_squared_error', optimizer=self.adam)
        self.Amodel.summary()
        self.model_copy = clone_model(self.Amodel)
    

        feature_input = Input(shape=(None, 144, 192, 3))
        xfe = TimeDistributed(Conv2D(100, (12, 12), activation='relu', padding='same'))(input_st)
        xfe = TimeDistributed(BatchNormalization())(xfe)
        xfe = TimeDistributed(Conv2D(80, (8, 8), activation='relu', padding='same'))(xfe)
        xfe = TimeDistributed(BatchNormalization())(xfe)
        xfe = TimeDistributed(Conv2D(80, (6, 6), activation='relu', padding='same'))(xfe)
        xfe = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(xfe)
        xfe = TimeDistributed(BatchNormalization())(xfe)
        xfe = TimeDistributed(Conv2D(60, (6, 6), activation='relu', padding='same'))(xfe)
        xfe = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(xfe)
        xfe = TimeDistributed(BatchNormalization())(xfe)
        xfe = TimeDistributed(Conv2D(60, (3, 4), activation='relu', padding='same'))(xfe)
        feature_output = TimeDistributed(Flatten())(xfe)
        print('feature_output.shape: ',feature_output.shape)
        self.femodel = Model(feature_input, feature_output)

        input_st0 = Input(shape=(None, 144, 192, 3), name='state')
        input_st1 = Input(shape=(None, 144, 192, 3), name='next_state')
        fout0 = self.feature(input_st0)
        fout1 = self.feature(input_st1, name=)
        inverse_input = concatenate([fout0,fout1])
        xi = TimeDistributed(Flatten())(inverse_input)
        xi = CuDNNLSTM(100, return_sequences=True )(xi)
        xi = TimeDistributed(Dense(100, activation='relu'))(x)
        inverse_output = TimeDistributed(Dense(8, activation='softmax'), name='inverse_output')(xi)
        print('inverse_output.shape: ',inverse_output.shape)
        self.imodel = Model([input_st0, input_st1], inverse_output([self.feature(input_st0), self.feature(input_st1)]))


        input_act = Input(shape=(None, 8), name='action')
        feature_shape = self.feature.layers[-1].output_shape
        print('feature_output.shape: ',feature_output.shape)
        input_from_femodel = Input(shape=feature_shape[1:])
        forward_input = concatenate([input_from_fmodel,input_act])
        print('forward_input: ', forward_input.shape)
        xfo = CuDNNLSTM(100, return_sequences=True )(forward_input)
        xfo = TimeDistributed(Dense(100, activation='relu'))(x)
        forward_output = TimeDistributed(Dense(feat_shape[2], activation='softmax'), name='forwarded_feature')(x)
        self.forward = Model(forward_input, forward_output)

        input_truef = Input(shape=(None, feat_shape[1:]))
        input_predf = Input(shape=(None, feat_shape[1:]))
        rew_out = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=0))([input_truef,input_act])
        self.rewint = Model([input_truef,input_predf],rew_out)

        def 

        self.icm = Model(inputs=[action, state, next_state], outputs=[forwarded_feature, action])


    

    

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



    def choose_action(self, state_col, step):
        if (np.random.random() <= self.get_epsilon(step)):
            order = random.randint(0,7)
        else:
            lat = self.encoder.predict(self.process_state_deque(state_col))
            zz = self.Amodel.predict(lat)
            print([round(zz[0,-1,i],6) for i in range(len(zz[0,-1,:]))], end= ', ')
            order = np.argmax(zz[0,-1,:])
            self.chosen = 1
        return self.convert_order(order)
        
    def wrapped_step(self, action):
        next_state, reward, done, info = self.env.step(action)
        resized = cv2.resize(next_state,(192,144))
        resized = np.reshape(np.array(resized / 255.0, dtype=np.float32), (144, 192, 3))
        return resized, reward, done, info

    def wrapped_reset(self):
        state = self.env.reset()
        resized = cv2.resize(state,(192,144))
        resized = np.reshape(np.array(resized / 255.0, dtype=np.float32), (144, 192, 3))
        return resized


    def get_epsilon(self, t):
        return self.epsilon_min + (self.epsilon_max - self.epsilon_min) * math.exp(-1 * t / self.epsilon_tau)

    def process_state_deque(self, state_deque):
        return np.reshape(np.array(state_deque), (1, len(state_deque), 144, 192, 3)).astype(np.float32)

    def process_memory(self, state, adescn, reward, next_state, done):
        self.state_small.append(state)
        nexts = self.state_small.copy()
        nexts.append(next_state)
        st = self.process_state_deque(self.state_small)
        lat = self.encoder.predict(st)
        next_st = self.process_state_deque(nexts)
        next_lat = self.encoder.predict(next_st)
        y_target = self.Amodel.predict(lat)
        y_target[0,-1,adescn] = reward if done else (reward + self.gamma * np.max(self.Bmodel.predict(next_lat)[0,-1,:]))
        self.xdata.append(np.copy(lat[-1,-1,:]))
        self.ydata.append(np.copy(y_target[-1,-1,:]))
        self.sdata.append(np.copy(state))
        if len(self.xdata) == self.state_len:
            self.docx.append(np.copy(self.xdata))
            self.docy.append(np.copy(self.ydata))
            self.docs.append(np.copy(self.sdata))
        if done:
            self.replay()

    def replay(self):
        #self.xdata.clear()
        #self.ydata.clear()
        self.docs = np.reshape(np.array(self.docs, dtype=np.float32), (len(self.docx), self.state_len, 144, 192, 3))
        self.docx = np.reshape(np.array(self.docx, dtype=np.float32), (len(self.docx), self.state_len, self.enc_shape[2], self.enc_shape[3], self.enc_shape[4]))
        self.docy = np.reshape(np.array(self.docy, dtype=np.float32), (len(self.docy), self.state_len, 8))
        
        self.Amodel.fit(self.docx, self.docy, batch_size = self.batchsize, verbose=1, epochs=1, shuffle=True)
        self.acoder.fit(self.docs, self.docs, batch_size = self.batchsize, verbose=1, epochs=1, shuffle=True)
        self.docx, self.docy, self.docs = [], [], []
        
        

    def run(self):
        self.scores = deque(maxlen=100)
        cur_mem = deque(maxlen = self.state_len)
        self.mean = np.mean(self.scores)
        self.steps = 0
        
        for e in range(self.n_episodes):
            state = self.wrapped_reset()
            cur_mem.append(np.copy(state))
            action, adesc, adescn = self.choose_action(cur_mem, self.steps+1)
            state1 = np.copy(state)
            adescn1 = adescn
            done = False
            i = 0
            episode_rew = 0
            tt = 0
            trew = 0
            time0 = time.time()
            xlast = 0
            self.flag = 0
            self.steps2 = 0
            while True:
                if (tt == 12) or done:
                    cur_mem.append(np.copy(state))
                    action, adesc, adescn = self.choose_action(cur_mem, self.steps+1)
                    next_state, reward, done, info = self.wrapped_step(action)
                    if xlast == info['x']:
                        self.flag += 1
                        if self.flag >= 5:
                            reward -= 5 * 12/36000
                    else:
                        self.flag = 0
                        xlast = info['x']
                    self.env.render()
                    state0 = np.copy(state1)
                    adescn0 = adescn1
                    state1 = np.copy(state)
                    adescn1 =adescn
                    self.process_memory(state0, adescn0, trew, state1, done)
                    trew = 0
                    tt = 0
                    self.steps += 1
                    self.steps2 += 1
                    if (self.steps2 == 502) and e>0:
                        self.predicted = self.acoder.predict(self.process_state_deque(cur_mem))
                        fig, axes = plt.subplots(1, self.state_len)
                        fig.set_size_inches(10, 10)
                        for im in range(self.state_len):
                            axes[im].imshow(self.predicted[-1,im,:])
                        plt.show(block=False)
                        plt.pause(3)
                        plt.close()
                        self.steps2 = 0
                    if done:
                        self.state_small.clear()
                        print(time.time()-time0)
                        cur_mem.clear()
                        self.scores.append(episode_rew)
                        self.mean = np.mean(self.scores)
                        self.model_copy.set_weights(self.Amodel.get_weights())
                        self.Amodel.set_weights(self.Bmodel.get_weights())
                        self.Bmodel.set_weights(self.model_copy.get_weights())
                        self.Amodel.compile(loss='mean_squared_error', optimizer=self.adam)
                        self.xdata.clear()
                        self.ydata.clear()
                        self.sdata.clear()
                        break
                else:
                    if adesc == 'B':
                        action, adesc, adescn = self.convert_order('pass')
                    else:
                        action, adesc, adescn = self.convert_order(adesc)
                    next_state, reward, done, info = self.wrapped_step(action)
                    #self.env.render()
                state = np.copy(next_state)
                tt += 1
                i += 1
                trew += reward
                episode_rew += reward
                
                if self.chosen == 1:
                    self.chosen = 0
                    print('step: {}, episode: , {}, episode_reward: , {:.6}, action: , {}, mean: {:.5}'.format(self.steps, e, episode_rew, adesc, self.mean))



if __name__ == '__main__':
    agent = SonicAgent()
    agent.run()










