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
from keras.models import Sequential
from keras.layers import Input, Dense, UpSampling2D
from keras.models import clone_model
from keras.layers import BatchNormalization
from keras.layers import TimeDistributed
from keras.layers import CuDNNLSTM
from keras.layers import Conv2D
from keras.layers import Concatenate
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import Lambda
from keras import losses
from keras import regularizers
from keras.models import Model
from keras import backend as K
from keras.losses import mean_squared_error
from keras.losses import categorical_crossentropy

class SonicAgent():
    def __init__(self, n_episodes=5000, max_env_steps=None, state_len=4, gamma=0.985, batchsize=1):
        self.state_len = state_len
        self.epsilon_max = 1
        self.batchsize = batchsize
        self.height = 150
        self.width = 200
        self.env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1', scenario='contest')
        self.env.reset()
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.beta = 0.1
        self.timedelay = 15
        self.A_memory = []
        self.R_memory = []
        self.state_memory = []
        #self.adam1 = Adam(lr=1e-6, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        #self.adam2 = Adam(lr=1e-6, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.adam1 = SGD(lr=1e-5, momentum=0.0, decay=0.0, nesterov=False)
        self.adam2 = SGD(lr=1e-5, momentum=0.0, decay=0.0, nesterov=False)
        self.model, self.icm, self.feature_model, self.ireward = self.build_models()
        self.compiled = False

    def build_models(self):

        def last_image(tensor):
            return tensor[:,-1,:]

        image_input = Input(shape=(self.state_len, self.height, self.width, 3))
        xf = TimeDistributed(Conv2D(90, (9, 9), activation='relu', padding='same'))(image_input)
        xf = TimeDistributed(MaxPooling2D((2,2)))(xf)
        xf = TimeDistributed(BatchNormalization())(xf)
        print(xf.shape)
        xf = TimeDistributed(Conv2D(60, (6, 6), activation='relu', padding='same'))(xf)
        xf = TimeDistributed(MaxPooling2D((2,2)))(xf)
        xf = TimeDistributed(BatchNormalization())(xf)
        print(xf.shape)
        xf = TimeDistributed(Conv2D(60, (5, 5), activation='relu', padding='same'))(xf)
        xf = TimeDistributed(MaxPooling2D((3,3)))(xf)
        features = TimeDistributed(Flatten())(xf)
        print('feature shape is: ', features.shape)
        feature_model = Model(image_input, features)	  

        state_current = Input(shape=(self.state_len, self.height, self.width, 3), name='state_current')
        state_next = Input(shape=(self.state_len, self.height, self.width, 3), name='state_next')
        feature_current = feature_model(state_current)
        last_feature = Lambda(last_image)(feature_current)
        feature_dimension = int(last_feature.shape[1])

        feature_next = feature_model(state_next)
        inverse_input = Concatenate()([feature_current,feature_next])
        xi = TimeDistributed(Dense(feature_dimension, activation='relu'))(inverse_input)
        xi = TimeDistributed(Dense(feature_dimension, activation='relu'))(xi)
        xi = CuDNNLSTM(50, return_sequences=False )(inverse_input)
        xi = Dense(50, activation='relu')(xi)
        inverse_output = Dense(8, activation='softmax', name='inverse_output')(xi)
        
        input_action = Input(shape=(8,), name='action')
        recurrent_branch = CuDNNLSTM(50, return_sequences=False)(feature_current)

        
        forward_input = Concatenate()([last_feature, input_action, recurrent_branch])
        
        xfo = Dense(feature_dimension, activation='relu')(forward_input)
        xfo = Dense(feature_dimension, activation='relu')(xfo)
        forward_output = Dense(feature_dimension, activation='relu')(xfo)
        last_feature_next = Lambda(last_image)(feature_next)

        icm = Model(inputs=[input_action, state_current, state_next], outputs=[inverse_output])
        def icm_loss(ytrue, ypred):
            return self.beta * K.mean(0.5 * K.square(forward_output - last_feature_next), axis=1) + (1 - self.beta) * K.categorical_crossentropy(ytrue, ypred)
        
        icm.compile(loss=icm_loss, optimizer=self.adam2, metrics=['accuracy'])
        
        ireward_output = Lambda(lambda x: K.mean(0.5 * K.square(x[0] - x[1]), axis=1))([forward_output,last_feature_next])					
        ireward = Model(inputs=[input_action, state_current, state_next], outputs=ireward_output)

        #main_input = Input(shape=(self.state_len, self.height, self.width, 3))
        x = TimeDistributed(Dense(feature_dimension, activation='relu'))(feature_current)
        x = TimeDistributed(Dense(feature_dimension, activation='relu'))(x)
        reward_input = Input(shape=(8,))
        x = Dense(50, activation='relu')(x)
        main_output = Dense(8, activation='softmax')(x)

        main_model = Model([state_current, reward_input], main_output)
        main_model.add_loss(self.sample_loss(main_output, reward_input))
        main_model.compile(optimizer=self.adam1)
        return main_model, icm, feature_model, ireward

    def sample_loss(self, model_output, reward_input):
        return -K.mean(K.sum(reward_input * K.log(model_output), axis=1), axis=0)

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
        #elif (order == 'LB') or (order == 8):
        #    LEFT = 1
        #    B = 1
        #    adesc = 'LB'
        #    adescn = 8
        #elif (order == 'RB') or (order == 9):
        #    RIGHT = 1
        #    B = 1
        #    adesc = 'RB'
        #    adescn = 9
        act = np.array([B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z])
        return act, adesc, adescn



    def choose_action(self, state_col):
        states = self.process_state_deque(state_col)
        zero_rew = np.zeros((1, 8))
        policy = self.model.predict([states, zero_rew])[-1]
        print([round(policy[i],6) for i in range(len(policy))],', ', len(self.state_memory))
        order = self.convert_order(np.random.choice(8, size=None, p=policy))
        return order
        
    def wrapped_step(self, action):
        next_state, reward, done, info = self.env.step(action)
        resized = cv2.resize(next_state,(self.width, self.height))
        resized = np.reshape(np.array(resized / 255.0, dtype=np.float32), (self.height, self.width, 3))
        return resized, reward, done, info

    def wrapped_reset(self):
        state = self.env.reset()
        resized = cv2.resize(state, (self.width,self.height))
        resized = np.reshape(np.array(resized / 255.0, dtype=np.float32), (self.height, self.width, 3))
        return resized

    def process_state_deque(self, state_deque):
        return np.reshape(np.array(state_deque, dtype=np.float32), (1, len(state_deque), self.height, self.width, 3))

    def replay(self):
        self.Rdata = []
        tr = 0
        steps = len(self.state_memory[1:])
        states=deque(maxlen=self.state_len)
        current = np.reshape(np.array(self.state_memory[0:-1]), (steps, self.state_len, self.height, self.width, 3)).astype(np.float32)
        next = np.reshape(np.array(self.state_memory[1:]), (steps, self.state_len, self.height, self.width, 3)).astype(np.float32)
        self.state_memory = []
        a = np.array(self.A_memory[:-1])
        self.A_memory = []
        print('a: ', a.shape)
        actions = np.zeros((steps,8))
        for ind in range(steps):
            actions[ind, a[ind]] = 1
        intrinsic = self.ireward.predict([actions, current, next], verbose=1, batch_size=5)
        self.R_memory = self.R_memory[:-1]
        tempr = []
        for (rewardi, rewarde) in zip(reversed(intrinsic), reversed(self.R_memory)):
            tr = 0.2 * rewarde + 0.8 * rewardi * self.timedelay / 36001 + tr * self.gamma
            tempr.append(tr)
        tempr.reverse()
        self.Rdata = np.copy(actions)
        for ind in range(steps):
            self.Rdata[ind] = self.Rdata[ind,:] * tempr[ind]

        for layer in self.feature_model.layers:
            layer.trainable = False
        
        self.Rdata = np.reshape(np.array(self.Rdata), (len(self.Rdata), 8)).astype(np.float32)
        print('--------------------------fit main model--------------------------------------')
        if not self.compiled:
            self.model.add_loss(self.sample_loss(self.model.output, self.model.inputs[1]))
            self.model.compile(optimizer=self.adam1)
            self.compiled = True
        self.model.fit(x=[current, self.Rdata], y=[], verbose=1, shuffle=True, batch_size=1)
        print('--------------------------fit icm--------------------------------------')
        for layer in self.feature_model.layers:
            layer.trainable = True
        self.icm.fit(x=[actions, current, next], y=[actions], verbose=1, shuffle=True, batch_size=1)
        

    def run(self):
        self.scores= []
        self.score100 = deque(maxlen = 100)
        self.mean_score = []
        episodes = []
        cur_mem = deque(maxlen = self.state_len)
        i = 0
        for e in range(self.n_episodes):
            episodes.append(e+1)
            for j in range(self.state_len):
                cur_mem.append(np.zeros((self.height, self.width, 3),dtype=np.float32))
            state = self.wrapped_reset()
            cur_mem.append(np.copy(state))
            action, adesc, adescn = self.choose_action(cur_mem)
            done = False
            self.substep = 0
            episode_reward = 0
            control_count = 0
            frame_reward = 0
            while True:
                if (control_count == self.timedelay) or done:
                    self.state_memory.append(np.copy(cur_mem))
                    self.A_memory.append(adescn)
                    self.R_memory.append(frame_reward)
                    if done:
                        cur_mem.clear()
                        self.scores.append(episode_reward)
                        self.score100.append(episode_reward)
                        self.mean_score.append(np.mean(self.score100))
                        print('episode: ', e, 'episode_reward: ', episode_reward, 'mean_reward: ', self.mean_score[-1])
                        self.replay()
                        
                        break
                    
                    action, adesc, adescn = self.choose_action(cur_mem)
                    next_state, reward, done, info = self.wrapped_step(action)
                    self.env.render()
                    frame_reward = 0
                    control_count = 0

                else:
                    action, adesc, adescn = self.convert_order(adesc)
                    next_state, reward, done, info = self.wrapped_step(action)
                #state = np.copy(next_state)
                if self.substep == 3:
                    cur_mem.append(np.copy(next_state))
                    self.substep = 0
                self.substep += 1
                control_count += 1
                frame_reward += reward
                episode_reward += reward
               



if __name__ == '__main__':
    agent = SonicAgent()
    agent.run()


        #   self.encoder, self.decoder, self.acoder = self.build_coders()
        #self.enc_shape = self.encoder.layers[-1].output_shape
        #print(self.enc_shape)


    #    # Init model
    #def build_coders(self):
    #    input_img = Input(shape=(None, self.height, self.width, 3))
    #    x = TimeDistributed(Conv2D(60, (6, 6), activation='relu', padding='same'))(input_img)
    #    x = TimeDistributed(BatchNormalization())(x)
    #    print(x.shape)
    #    x = TimeDistributed(Conv2D(60, (6, 6), activation='relu', padding='same'))(x)
    #    x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)
    #    x = TimeDistributed(BatchNormalization())(x)
    #    print(x.shape)
    #    x = TimeDistributed(Conv2D(40, (4, 4), activation='relu', padding='same'))(x)
    #    x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)
    #    x = TimeDistributed(BatchNormalization())(x)
    #    print(x.shape)
    #    x = TimeDistributed(Conv2D(40, (4, 4), activation='relu', padding='same'))(x)
    #    encoded = TimeDistributed(MaxPooling2D((3, 3), padding='same'))(x)
    #    encoder = Model(input_img, encoded)

    #    enc_shape = encoder.layers[-1].output_shape
    #    print(enc_shape[1:])

    #    input_encoded = Input(shape=enc_shape[1:])
    #    x = TimeDistributed(BatchNormalization())(input_encoded)
    #    x = TimeDistributed(Conv2D(40, (4, 4), activation='relu', padding='same'))(x)
    #    x = TimeDistributed(UpSampling2D((3, 3)))(x)
    #    x = TimeDistributed(BatchNormalization())(x)
    #    print(x.shape)

    #    x = TimeDistributed(Conv2D(60, (4, 4), activation='relu', padding='same'))(x)
    #    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    #    x = TimeDistributed(BatchNormalization())(x)
    #    print(x.shape)

    #    x = TimeDistributed(Conv2D(60, (6, 6), activation='relu', padding='same'))(x)
    #    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    #    x = TimeDistributed(BatchNormalization())(x)
    #    print(x.shape)
    #    decoded = TimeDistributed(Conv2D(3, (6, 6), activation='relu', padding='same'))(x)
        
    #    print(decoded.shape)
    #    decoder = Model(input_encoded, decoded)
    #    acoder = Model(input_img, decoder(encoder(input_img)))
    #    #adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    #    sgd = SGD(lr=0.001, momentum=0.1, decay=0.0, nesterov=False)
    #    acoder.compile(loss='mean_squared_error', optimizer=sgd)
    #    encoder.compile(loss='mean_squared_error', optimizer=sgd)
    #    decoder.compile(loss='mean_squared_error', optimizer=sgd)
    #    print('---------------------------------------')
    #    acoder.summary()
    #    print('---------------------------------------')
    #    return encoder, decoder, acoder

