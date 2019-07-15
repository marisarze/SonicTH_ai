
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
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
from keras.models import load_model
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
from keras.layers import add
from keras.layers import Activation
from keras.layers import Lambda
from keras.layers import average
from keras.layers import PReLU
from keras.layers import LeakyReLU
from keras import losses
from keras import regularizers
from keras.models import Model
from keras import backend as K
from keras.losses import mean_squared_error
from keras.losses import categorical_crossentropy
from threading import Thread
import tensorflow as tf
import queue
from retrowrapper import RetroWrapper
K.set_epsilon(1e-7)
K.set_floatx('float32')

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

class SonicAgent():
    def __init__(self, n_episodes=50000, max_env_steps=None, state_len=1, gamma=0.99, batch_size=32):
        self.usb = False
        self.crange = 0.1
        self.bm = 0.999
        self.action_space = 10
        self.state_len = state_len
        self.epsilon_max = 1
        self.batch_size = batch_size
        self.height = 90
        self.width = 128
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.beta = 0.1
        self.timedelay = 10
        self.V = []
        self.old = []
        self.tau = 50000
        self.default = np.float32
        #self.adam1 = Adam(lr=5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-5, decay=0.0, amsgrad=False, clipvalue=5.0)
        #self.adam2 = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0, clipnorm=10., amsgrad=False)
        self.adam1 = SGD(lr=2e-4, momentum=0.0, decay=0.0, nesterov=False, clipvalue=50.0)
        #self.adam2 = SGD(lr=1e-2, momentum=0.0, decay=0.0, nesterov=False)
        #self.model_copy = clone_model(self.Amodel)
        self.compiled = False
        self.steps = 0 
        self.steps2 = 0
        self.training_steps = 0
        self.rcheck = 0
        self.adv = np.float32
        self.state_buffer = []
        self.advantage_buffer = []
        self.value_buffer = []
        self.old_buffer = []
    def build_model(self):
        def last_image(tensor):
            return tensor[:,-1,:]
        def crelu(x):
            return K.relu(x, alpha=0.0, max_value=1, threshold=0.0)
        main_input = Input(shape=(self.state_len, self.height, self.width, 3))
        xm = TimeDistributed(Conv2D(40, (8,8), padding='same', use_bias=self.usb))(main_input)
        xm = TimeDistributed(BatchNormalization(momentum=self.bm))(xm)
        #xm = TimeDistributed(Activation('tanh'))(xm)
        xm = TimeDistributed(PReLU())(xm)
        xm = TimeDistributed(MaxPooling2D((4,4)))(xm)
        xm = TimeDistributed(Conv2D(80, (6,6), padding='same', use_bias=self.usb))(xm)
        xm = TimeDistributed(BatchNormalization(momentum=self.bm))(xm)
        #xm = TimeDistributed(Activation('tanh'))(xm)
        xm = TimeDistributed(PReLU())(xm)
        xm = TimeDistributed(MaxPooling2D((4,4)))(xm)
        xm = TimeDistributed(Conv2D(120, (6,6), padding='same', use_bias=self.usb))(xm)
        xm = TimeDistributed(BatchNormalization(momentum=self.bm))(xm)
        #xm = TimeDistributed(Activation('tanh'))(xm)
        xm = TimeDistributed(PReLU())(xm)
        xm = TimeDistributed(MaxPooling2D((3,3)))(xm)
        xm = TimeDistributed(Flatten())(xm)
        #xmr = CuDNNLSTM(50, return_sequences=False)(xm)
        #xm = TimeDistributed(BatchNormalization())(xm)
        #xmr = CuDNNLSTM(10, return_sequences=False)(xm)
        
        xm1= Lambda(last_image)(xm)

        #features = Concatenate()([xm1,xmr])
        features = BatchNormalization()(xm1)
        feature_dimension = int(features.shape[1])
        print('feature shape is: ', feature_dimension)
        
        xm = Dense(feature_dimension, use_bias=self.usb)(features)
        xm = BatchNormalization(momentum=self.bm)(xm)
        #xm = Activation('tanh')(xm)
        xm = PReLU()(xm)
        #xm = BatchNormalization()(xm)
        #feature_dimension = math.ceil(feature_dimension/2)
        xm = Dense(feature_dimension, use_bias=self.usb)(xm)
        xm = BatchNormalization()(xm)
        xm = PReLU()(xm)
        reward_input = Input(shape=(self.action_space,))
        old_input = Input(shape=(self.action_space,))
        #xm = BatchNormalization()(xm)
        xm = Dense(self.action_space,  use_bias=self.usb)(xm)
        xm = BatchNormalization(momentum=self.bm)(xm)
        main_output = Activation('softmax', name='main_output')(xm)

        #xc = BatchNormalization()(xm1)
        xc = Dense(feature_dimension, use_bias=self.usb)(features)
        xc = BatchNormalization(momentum=self.bm)(xc)
        #xc = Activation('tanh')(xc)
        xc = PReLU()(xc)
        
        xc = Dense(1, use_bias=self.usb)(xc)
        xc = BatchNormalization(momentum=self.bm)(xc)
        critic_output = Activation('tanh', name='critic_output')(xc)
        #critic_output = PReLU(name='critic_output')(xc)
        model = Model(inputs=[main_input,reward_input,old_input],outputs=[main_output, critic_output])
        model.compile(optimizer=self.adam1, 
                        loss={'main_output': self.sample_loss(reward_input,old_input), 'critic_output': self.critic_loss}, 
                        loss_weights={'main_output': 1, 'critic_output': 1})
        exists = os.path.isfile('model.h5')
        if exists:
            model.load_weights('model.h5')
            print('model weights loaded')
        #model.summary()
        return model

    def sample_loss(self, reward_input, old_input):
        def custom_loss(y_true, y_pred):
            ratio = y_pred/(old_input+1e-4)
            pg_loss1 = -reward_input * ratio
            pg_loss2 = -reward_input * K.clip(ratio, 1.0 - self.crange, 1.0 + self.crange)
            pg_loss = K.max((pg_loss1,pg_loss2),axis=0)
            return K.mean(K.mean(pg_loss,axis=1),axis=0)
            #return -K.mean(K.sum(reward_input * K.log(y_pred+1e-3), axis=1), axis=0)# + K.mean(K.sum(0.05 * K.square(y_pred/(old_input)-1),axis=1),axis=0)
        return custom_loss
    def critic_loss(self, y_true, y_pred):
        return K.mean(0.5 * K.square(y_true - y_pred), axis=-1)

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
        elif (order == 'LB') or (order == 8):
            LEFT = 1
            B = 1
            adesc = 'LB'
            adescn = 8
        elif (order == 'RB') or (order == 9):
            RIGHT = 1
            B = 1
            adesc = 'RB'
            adescn = 9
        act = np.array([B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z])
        return act, adesc, adescn



    def choose_action(self, state_col, Amodel):
        states = np.reshape(np.array(state_col)/127-1, (1, len(state_col), self.height, self.width, 3)).astype(self.default)
        zero_rew = np.zeros((1, self.action_space))
        zero_old = np.zeros((1, self.action_space))
        policy, value = Amodel.predict([states, zero_rew, zero_old])
        policy = policy[-1]
        #print([round(policy[i],6) for i in range(len(policy))], 'steps: ', self.steps, 'value: ', repr(V[0,0]), 'buflen: ', len(self.value_buffer))
        if random.random() < 0.05:#0.05 + 0.95*math.exp(-self.training_steps / self.tau):
            order = random.randint(0,9)
            #print('random: ', order, 'steps: ', self.steps)
            order = self.convert_order(order)
        else:
            order = self.convert_order(np.random.choice(self.action_space, size=None, p=policy))
        self.steps += 1
        self.steps2 += 1
        #print(order, policy, value[-1])
        return order, policy, value[-1]
        
    def wrapped_step(self, action, env):
        next_state, reward, done, info = env.step(action)
        resized = cv2.resize(next_state,(self.width, self.height))
        resized = np.array(resized, dtype=np.uint8)
        return resized, reward, done, info

    def wrapped_reset(self, env):
        state = env.reset()
        resized = cv2.resize(state, (self.width,self.height))
        resized = np.array(resized, dtype=np.uint8) 
        return resized

    def compute_advantages(self, reward_memory, action_memory, value_memory):
        #reward_memory = np.array(self.reward_memory).astype(np.float64)
        #sumi = np.sum(reward_memory)
        #if sumi:
        #    maxi = np.max(np.nonzero(reward_memory))
        #    reward_memory.fill(0)
        #    reward_memory[maxi]= sumi
        G = []
        tr = 0
        for reward in reversed(reward_memory):
            tr = reward + tr * self.gamma
            G.append(tr)
        G.reverse()
        G = np.array(G).astype(np.float64)
        #G -= np.mean(G)
        #G /= np.std(G)
        steps = len(value_memory)
        actions = np.zeros((steps,self.action_space))
        Advantage = np.zeros((steps,self.action_space)).astype(self.adv)
        V_target = np.zeros((steps,1)).astype(self.adv)
        td = np.zeros((steps,1)).astype(self.adv)
        multistep = 10
        
        for ind in range(steps):
            #if ind+multistep < steps:
            #    multirew = 0
            #    for rew in reversed(reward_memory[ind:ind+multistep]):
            #        multirew = rew + self.gamma * multirew
            #    td[ind] = multirew + self.gamma * V_next[ind+multistep-1] - V_current[ind]
            #    V_target[ind,0] = multirew + self.gamma * V_next[ind+multistep-1]
            #else:
            td[ind] = G[ind] - value_memory[ind]
            V_target[ind,0] = G[ind]
            Advantage[ind, action_memory[ind]] = td[ind]

        return Advantage, V_target
        
        #td[ind] = self.R_memory[ind]
        #Advantage[ind, a[ind]] = td[ind]
        #Advantage -= np.mean(Advantage)
        #Advantage /= np.std(Advantage)



    
    def batch_generator(self):
        while True:
            self.state_buffer = np.array(self.state_buffer, dtype=np.uint8)
            self.advantage_buffer = np.array(self.advantage_buffer)
            self.value_buffer = np.array(self.value_buffer)
            self.old_buffer = np.array(self.old_buffer)
            randomize = np.arange(len(self.state_buffer))
            np.random.shuffle(randomize)
            print('st_buf:',len(self.state_buffer))
            print('advantage_buf:',len(self.advantage_buffer))
            print('value_buf:',len(self.value_buffer))
            print('old_buf:',len(self.old_buffer))
            self.state_buffer = self.state_buffer[randomize]
            self.advantage_buffer = self.advantage_buffer[randomize]
            self.value_buffer = self.value_buffer[randomize]
            self.old_buffer = self.old_buffer[randomize]
            for ind in range(0, len(self.value_buffer)-self.batch_size, self.batch_size):
                state_batch = np.array(self.state_buffer[ind:ind+self.batch_size,:], dtype=np.float32)/127-1
                advantage_batch = np.array(self.advantage_buffer[ind:ind+self.batch_size,:], dtype=np.float32)
                old_batch = np.array(self.old_buffer[ind:ind+self.batch_size,:], dtype=np.float32)
                value_batch = np.array(self.value_buffer[ind:ind+self.batch_size], dtype=np.float32)
                zero_action_batch = np.zeros((self.batch_size,self.action_space))
                yield [state_batch, advantage_batch,  old_batch], [zero_action_batch,  value_batch]
        
    def run_episode(self, episode, Amodel, render=False):
        env = RetroWrapper(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1', scenario='contest')
        cur_mem = deque(maxlen = self.state_len)
        done = False
        self.substep = 0
        episode_reward = 0
        control_count = 0
        frame_reward = 0
        state_memory = []
        reward_memory = []
        action_memory = []
        old_memory = []
        value_memory = []
        for j in range(self.state_len):
            cur_mem.append(np.zeros((self.height, self.width, 3),dtype=self.default))
        state = self.wrapped_reset(env)
        cur_mem.append(np.copy(state))
        state_memory.append(np.copy(cur_mem))
        (action, adesc, adescn), cur_policy, cur_value = self.choose_action(cur_mem, Amodel)
        old_memory.append(cur_policy)
        value_memory.append(cur_value)
        while True:
            if (control_count == self.timedelay) or done:
                state_memory.append(np.copy(cur_mem))
                action_memory.append(adescn)
                reward_memory.append(frame_reward)
                if done:
                    self.scores.append(episode_reward)
                    self.score100.append(episode_reward)
                    self.mean_score.append(np.mean(self.score100))
                    print('episode: ', len(self.scores), 'episode_reward: ', episode_reward, 'mean_reward: ', self.mean_score[-1])
                    return state_memory, reward_memory, action_memory, old_memory, value_memory
                    
                    
                (action, adesc, adescn), cur_policy, cur_value = self.choose_action(cur_mem, Amodel)
                old_memory.append(cur_policy)
                value_memory.append(cur_value)
                next_state, reward, done, info = self.wrapped_step(action, env)
                if render:
                    env.render()
                frame_reward = 0
                control_count = 0
            else:
                action, adesc, adescn = self.convert_order(adesc)
                next_state, reward, done, info = self.wrapped_step(action, env)
            #state = np.copy(next_state)
            if self.substep == 4:
                cur_mem.append(np.copy(next_state))
                self.substep = 0
            self.substep += 1
            control_count += 1
            frame_reward += reward
            episode_reward += reward

    def run_train(self):
        num_worker_threads = 8
        self.scores= []
        self.score100 = deque(maxlen = 100)
        self.mean_score = []
        episodes = []
        self.big_data1 = deque(maxlen = 100)
        self.big_data2 = deque(maxlen = 100)
        def worker1():
            with tf.Session(graph=tf.Graph(), config=tf.ConfigProto(
                    intra_op_parallelism_threads=8)) as sess:
                Amodel = self.build_model()
                K.set_session(sess)
                while True:
                    item = q.get()
                    if item is None:
                        break
                    render = False
                    if item == 0:
                        render = True
                    
                    state_memory, reward_memory, action_memory, old_memory, value_memory = self.run_episode(item, Amodel, render)
                    print(np.std(np.array(old_memory), axis=0)/np.mean(np.array(old_memory), axis=0))
                    advantage_memory, target_values = self.compute_advantages(reward_memory, action_memory, value_memory)
                    self.big_data2.append((state_memory[:-1], advantage_memory, target_values, old_memory))
                    print('----------------------------------------------------------------------------------------------------')
                    q.task_done()
                    

        while True:

            q = queue.Queue()
            threads = []
            for i in range(num_worker_threads):
                t = ThreadWithReturnValue(target=worker1)
                t.start()
                threads.append(t)
            for item in range(100):
                q.put(item)
            # block until all tasks are done
            q.join()
            # stop workers
            for i in range(num_worker_threads):
                q.put(None)
            for t in threads:
                t.join()
            
            self.state_buffer = list()
            self.advantage_buffer = list()
            self.value_buffer = list()
            self.old_buffer = list()

            while self.big_data2:
                (states, advantages, values, olds) = self.big_data2.popleft()
                self.state_buffer.extend(states)
                self.advantage_buffer.extend(advantages)
                self.value_buffer.extend(values)
                self.old_buffer.extend(olds)
        
            session = K.get_session()
            with tf.Session() as sess:
                K.set_session(sess)
                sess.run(tf.global_variables_initializer())
                graph = tf.get_default_graph()
                with graph.as_default():
                    steps = math.ceil(len(self.value_buffer)/self.batch_size)-1
                    Amodel = self.build_model()
                    Amodel.fit_generator(generator=self.batch_generator(), 
                                                steps_per_epoch=steps, 
                                                epochs=10, 
                                                verbose=1)
                    Amodel.save_weights('model.h5')

            
               



if __name__ == '__main__':
    agent = SonicAgent()
    agent.run_train()



         

    
        #print('--------------------------fit main model--------------------------------------')
        #prior_state = []
        #prior_A = []
        #prior_Vtar = []
        #prior_old = []
        #p_small = np.array([(1/(i+1))**0 for i in range(steps)])
        #p_big = p_small/np.sum(p_small)
        
        #indes = np.argsort(np.absolute(td[:,0]))[::-1]
        ##wait = input("PRESS ENTER TO CONTINUE.")
        #coeff = self.scores[-1]
        #allen =math.ceil(50 * steps * coeff/self.batchsize) * self.batchsize
        #w = 1 / allen / p_big
        #w /= np.max(w)
        #len2 = 0
        #prior_state = current
        #prior_A = Advantage
        #prior_Vtar = V_target
        #prior_old = self.old
        #self.Bmodel.fit(x=[prior_state[:-1,:], prior_A, prior_old[:-1,:]], 
        #                y=[np.zeros((steps,self.action_space)), prior_Vtar], 
        #                verbose=1, 
        #                shuffle=True, 
        #                batch_size=self.batchsize)
        #while len2 < allen:
        #    ind = np.random.choice(steps, size=None, p=p_big)
        #    ind = indes[ind]
        #    prior_state.append(current[ind])
        #    prior_A.append(w[ind] * Advantage[ind,:])
        #    prior_Vtar.append(V_target[ind,:])
        #    prior_old.append(self.old[ind,:])
        #    priorl = len(prior_state)
        #    if priorl == min(4*self.batchsize, allen-len2):
        #        prior_state = np.reshape(np.array(prior_state), (priorl, self.state_len, self.height, self.width, 3)).astype(self.default)
        #        prior_A = np.reshape(np.array(prior_A), (priorl, self.action_space)).astype(self.adv)
        #        prior_Vtar = np.reshape(np.array(prior_Vtar), (priorl, 1)).astype(self.adv)
        #        prior_old = np.reshape(np.array(prior_old), (priorl, self.action_space)).astype(self.default)
        #        self.Amodel.fit(x=[prior_state, prior_A, prior_old], 
        #                        y=[np.zeros((priorl,self.action_space)), prior_Vtar], 
        #                        verbose=1, 
        #                        shuffle=True, 
        #                        batch_size=self.batchsize)
        #        len2 += priorl
        #        self.training_steps += priorl
        #        print('tr_steps: ', self.training_steps)
        #        print('cur_tr_steps: ', priorl)
        #        prior_state = []
        #        prior_A = []
        #        prior_Vtar = []
        #        prior_old = []
        #if self.rcheck == 10:
        #    #self.model_copy.set_weights(self.Amodel.get_weights())
        #    self.Amodel.set_weights(self.Bmodel.get_weights())
        #    #self.Bmodel.set_weights(self.model_copy.get_weights())
        #    self.rcheck = 0
        ##lr=max(1e-6, 1/(1000+self.steps))
        ##print('learning rate: ', lr)
        ##K.set_value(self.model.optimizer.lr, lr)
        #for ind in range(steps):
        #    if self.counts[ind]>=100:
        #        self.Gmean[ind]=(G[ind]+99*self.Gmean[ind])/100
        #    else:
        #        self.Gmean[ind]=(G[ind]+self.counts[ind]*self.Gmean[ind])/(self.counts[ind]+1)
        #        self.counts[ind] += 1
        #if self.scores[-1] <= self.mean_score[-1]:
        #    self.rcheck += 1
        #plt.ion()
        
        #plt.plot(self.Gmean, 'r')
        #plt.show(block=False)
        #plt.pause(0.001)
        #time.sleep(4)
        #plt.close('all')