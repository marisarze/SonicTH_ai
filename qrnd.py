
import os
import sys
import json
import random
import math
import numpy as np
from decimal import Decimal
import random
import time
import cv2
from decimal import Decimal
from collections import deque
from matplotlib import pyplot as plt
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import queue
from multiprocessing import Queue, Process, Lock

from keras.models import load_model
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.models import clone_model
from keras.layers import BatchNormalization
from keras.layers import TimeDistributed
from keras.layers import CuDNNLSTM
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Concatenate
from keras.layers import Reshape
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import UpSampling2D
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

def create_reward_net(state_len, fast=True):
    height = 84
    width = 120
    activ = LeakyReLU(alpha=0.3)

    def loss_function(x1,x2):
        def custom_loss(y_true, y_pred):
            return K.mean(K.pow(0.5 * (x1 - x2), 2), axis=-1)
        return custom_loss

    def last_image(tensor):
        return tensor[:,-1,:]

    def ireward(tensor):
        return  K.mean(K.pow(0.5 * (tensor[0] - tensor[1]), 2), axis=-1)

    state_input = Input(shape=(state_len, height, width, 3))
    float_input = K.cast(state_input, dtype='float32')
    float_input = Lambda(lambda input: input/255.0-0.5)(float_input)
    new_input = Lambda(last_image)(float_input)
    xs = Conv2D(32, (4,4), strides=(2,2), padding='same')(new_input)
    xs = activ(xs)
    xs = Conv2D(64, (4,4), strides=(2,2), padding='same')(xs)
    xs = activ(xs)
    xs = Conv2D(64, (4,4), strides=(2,2), padding='same')(xs)
    xs = activ(xs)
    xs = Conv2D(128, (4,4), strides=(2,2), padding='same')(xs)
    xs = activ(xs)
    stochastic_output = Flatten()(xs)
    stochastic_part = Model(inputs=state_input, outputs=stochastic_output)
    for layer in stochastic_part.layers:
        layer.trainable = False

    xt = Conv2D(32, (4,4), strides=(2,2), padding='same')(new_input)
    xt = activ(xt)
    xt = Conv2D(64, (4,4), strides=(2,2), padding='same')(xt)
    xt = activ(xt)
    xt = Conv2D(64, (4,4), strides=(2,2), padding='same')(xt)
    xt = activ(xt)
    xt = Conv2D(128, (4,4), strides=(2,2), padding='same')(xt)
    xt = activ(xt)
    target_output = Flatten()(xt)

    intrinsic_reward = Lambda(ireward)([stochastic_output, target_output])
    model = Model(inputs=state_input, outputs=intrinsic_reward)
    if fast:
        #adam = Adam(lr=2e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-5, decay=0.0, amsgrad=False)
        adam = SGD(lr= 2e-4, momentum=0)
    else:
        #adam = Adam(lr=2e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-5, decay=0.0, amsgrad=False)
        adam = SGD(lr= 2e-4, momentum=0)
    model.compile(optimizer=adam, loss=loss_function(target_output, stochastic_output))
    return model

def create_policy_net(model_type, state_len, action_space):
    height = 84
    width = 120
    beta = 0.1
    crange = 0.1
    activ1 = 'tanh'
    activ2 = LeakyReLU(alpha=0.2)
    def sample_loss(reward_input, old_input):
        def custom_loss(y_true, y_pred):
            entropy = -K.sum(y_pred * K.log(y_pred), axis=-1) / K.log(tf.constant(action_space, tf.float32))
            
            ratio = y_pred/(old_input+1e-3)
            pg_loss1 = -reward_input * ratio
            # entropy_old = -K.sum(old_input * K.log(old_input), axis=-1) / K.log(tf.constant(action_space, tf.float32))
            # d = K.pow(entropy_old, 1) * crange
            # d = tf.reshape(d, [tf.shape(d)[0],1])
            d = crange
            pg_loss2 =-reward_input * K.clip(ratio, 1-d, 1+d)
            pg_loss = K.maximum(pg_loss1,pg_loss2)
            loss = K.sum(pg_loss,axis=-1) + beta * K.abs(K.sum(reward_input,axis=-1)) * K.pow(entropy-1, 4)
            return loss
        return custom_loss

    def critic_loss(y_true, y_pred):
        return K.mean(0.5 * K.square(y_true - y_pred), axis=-1)
    def last_image(tensor):
        return tensor[:,-1,:]
    
    reward_input = Input(shape=(action_space,))
    old_policy_input = Input(shape=(action_space,))
    state_input = Input(shape=(state_len, height, width, 3))
    float_input = K.cast(state_input, dtype='float32')
    float_input = Lambda(last_image)(float_input)
    float_input = Lambda(lambda input: input/255.0-0.5)(float_input)
    x = Conv2D(32, (8,8), strides=(4,4), padding='same')(float_input)
    x = activ2(x)
    x = Conv2D(64, (4,4), strides=(2,2), padding='same')(x)
    x = activ2(x)
    x = Conv2D(128, (3,3), padding='same')(x)
    x = activ2(x)
    x = Flatten()(x)
    x = Dense(512, activation=activ1)(x)
    x = Dense(512, activation=activ1)(x)
    if model_type=='policy':
        #adam1 = Adam(lr=2e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-5, decay=0.0, amsgrad=False, clipnorm=1.0)
        adam1 = SGD(lr= 2e-4, momentum=0, clipnorm=1.0)
        main_output = Dense(action_space, activation='softmax', name='main_output')(x)
        model = Model(inputs=[state_input,reward_input,old_policy_input],outputs=main_output)
        model.compile(optimizer=adam1, loss=sample_loss(reward_input,old_policy_input))
    elif model_type=='critic':
        #adam1 = Adam(lr=2e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-5, decay=0.0, amsgrad=False)
        adam1 = SGD(lr= 2e-4, momentum=0, clipnorm=1.0)
        critic_output = Dense(1, activation='linear', name='critic_output')(x)
        model = Model(inputs=state_input,outputs=critic_output)
        model.compile(optimizer=adam1, loss=critic_loss)
    else:
        raise Exception('model type is not named')
    return model
    

class SonicAgent():
    def __init__(self, episodes_step=10, max_env_steps=None, state_len=1, gamma=0.999, batch_size=512, workers=6, render=False):
        self.timedelay = 15
        self.batch_size = batch_size
        self.csteps = episodes_step * 36000 / self.timedelay / self.batch_size
        self.workers = workers
        self.iw = 1
        self.ew = 0
        self.epochs = 4
        
        self.actions = self.get_actions()
        self.action_space = len(self.actions)
        self.state_len = 1
        self.height = 84
        self.width = 120
        self.lam = 0.95
        self.gamma = gamma
        self.egamma = gamma
        self.igamma = gamma
        self.ignored_steps = 1
        self.cutted_steps = max(self.ignored_steps, 15)
        self.render = render
        self.default = np.float32
        self.adv = np.float32
        self.stats = dict()
        self.maxstat = 3000

        self.critical_steps = 3600
        self.choosed_length = 4 * self.critical_steps
        self.minimal_steps = 2 * self.choosed_length
        self.horizon = 2 * self.choosed_length
        # self.special_game_ids = [2**i for i in range(20)]
        # temp = []
        # for num in self.special_game_ids[:]:
        #     for i in range(0,4):
        #         temp.append(num+i)
        # self.special_game_ids = temp[:]
        


    def get_entropy(self, policy_sequence):
        policy_array = np.array(policy_sequence)
        entropy = -np.sum(policy_array * np.log(policy_array), axis=-1) / np.log(self.action_space)
        return entropy
    
    def get_imodel_result(self, state_memory, imodel_new, imodel_old, entropy):
        states = self.process_states(state_memory)
        iloss = imodel_new.predict(states)
        ratio = iloss 
        if self.stats['initialized'] > 0:
            ratio -= self.stats['imean']
            ratio /= self.stats['istd']
        
        irewards = self.iw * (1-self.igamma) * ratio

        return irewards, ratio

    def save_stats(self, path=r'D:\sonic_models\results.json'):
        try:
            with open(path, 'w') as file:
                json.dump(self.stats, file, indent=4)
            print('Stats saved')
        except:
            print('Error occured for saving stats')

    def run_train(self):
        lock = Lock()
        self.get_stats()
        
        lock.acquire()
        Amodel = create_policy_net('policy', self.state_len, self.action_space)
        if os.path.isfile(r'D:\sonic_models\policy_model.h5'):
            Amodel.load_weights(r'D:\sonic_models\policy_model.h5')
        eVmodel = create_policy_net('critic', self.state_len, self.action_space)
        if os.path.isfile(r'D:\sonic_models\evalue_model.h5'):
            eVmodel.load_weights(r'D:\sonic_models\evalue_model.h5')

        iVmodel = create_policy_net('critic', self.state_len, self.action_space)
        if os.path.isfile(r'D:\sonic_models\ivalue_model.h5'):
            iVmodel.load_weights(r'D:\sonic_models\ivalue_model.h5')

        imodel_new = create_reward_net(self.state_len)
        if os.path.isfile(r'D:\sonic_models\imodel_new.h5'):
            imodel_new.load_weights(r'D:\sonic_models\imodel_new.h5')

        imodel_old = create_reward_net(self.state_len, fast=False)
        if os.path.isfile(r'D:\sonic_models\imodel_old.h5'):
            imodel_old.load_weights(r'D:\sonic_models\imodel_old.h5')

        lock.release()
        num_workers = self.workers
        data = Queue()
        task_sequence = Queue()
        new_weights = Queue()
        render_que = Queue()
        #render_que.put(True)
        processes = []
        for i in range(num_workers):
            processes.append(Process(target=self.worker, args=(lock, data, task_sequence, render_que)))
            processes[-1].start()
        print(processes)
        
        #render_que.put(True)

        for game_id in range(self.stats['episodes_passed'] + 1, 5000000):
            task_sequence.put(game_id)
        state_buffer = list()
        advantage_buffer = list()
        evalue_buffer = list()
        ivalue_buffer = list()
        old_buffer = list()
        state_memory = list()
        #eadvantage_memory = list()
        iadvantage_memory = list()
        advantage_memory = list()
        #evalue_memory = list()
        ivalue_memory = list()
        old_memory = list()
        action_memory = list()
        ireward_memory = list()
        #ereward_memory = list()
        excluded = ['x', 'y', 'some_map', 'ireward_map', 'entropy_map', 'cutted']
        for elem in excluded:
            self.stats[elem] = []
        print('---------------------------------------------------------')
        count = 0
        while True:
            if data.qsize() >= 1:
                count += 1
                (states, erewards, actions, olds, coords) = data.get()
                steps = len(erewards)
                entropy = self.get_entropy(olds)
                self.stats['action_std'].append(np.std(olds))
                self.stats['episodes_passed'] += 1
                self.stats['episodes_numbers'].append(self.stats['episodes_passed'])
                self.stats['steps'] += steps
                self.stats['steps_list'].append(self.stats['steps']) 
                self.stats['emean'] = ((self.horizon - steps) * self.stats['emean'] + steps * np.mean(entropy))/self.horizon
                self.stats['entropy'].append(self.stats['emean'])
                self.stats['mean100_entropy'].append(np.mean(self.stats['entropy'][-100:]))
                self.stats['external_rewards'].append(np.sum(erewards))
                self.stats['external_rewards_per_step'].append(np.sum(erewards)/steps)
                self.stats['mean100_external_rewards'].append(np.mean(self.stats['external_rewards'][-100:]))
                irewards, ratio = self.get_imodel_result(states, imodel_new, imodel_old, entropy)

                self.stats['x'].extend(coords['x'][:-self.cutted_steps])
                self.stats['y'].extend(coords['y'][:-self.cutted_steps])
                self.stats['some_map'].extend(list(ratio[:-self.cutted_steps]))
                self.stats['ireward_map'].extend(list(irewards[:-self.cutted_steps]))
                self.stats['entropy_map'].extend(list(entropy[:-self.cutted_steps]))
                self.stats['cutted'].append(steps-self.cutted_steps)

                print('episode: ', self.stats['episodes_passed'],
                    'episode_reward: ', self.stats['external_rewards'][-1],
                    'mean_reward: ', self.stats['mean100_external_rewards'][-1])
                print('mean_entropy:', self.stats['emean'])
                print('sum_irewards:', np.sum(irewards))
                print('steps:', steps)
                print('std:', np.std(olds, axis=0))
                
                # eadvantages, etargets = self.compute_advantages(states,
                #                                                 erewards,
                #                                                 actions,
                #                                                 eVmodel,
                #                                                 self.egamma,
                #                                                 self.lam,
                #                                                 True,
                #                                                 False)
                iadvantages, itargets = self.compute_advantages(states,
                                                                irewards,
                                                                actions,
                                                                iVmodel,
                                                                self.igamma,
                                                                self.lam,
                                                                True,
                                                                False)

                state_memory.extend(states)
                ireward_memory.extend(irewards)
                #ereward_memory.extend(erewards)
                action_memory.extend(actions)
                old_memory.extend(olds)
                #eadvantage_memory.extend(eadvantages)
                iadvantage_memory.extend(iadvantages)
                #evalue_memory.extend(etargets)
                ivalue_memory.extend(itargets)
                print('---------------------------------------------------------')
            if (len(state_memory) > self.choosed_length) and count>32:

                count = 0
                advantage_memory = np.array(iadvantage_memory)
                state_buffer = np.array(state_memory, dtype=np.uint8)
                advantage_buffer = np.array(advantage_memory)
                astd = np.std(np.sum(advantage_buffer, axis=-1))
                advantage_buffer /= astd
                
                #evalue_buffer = np.array(evalue_memory)
                ivalue_buffer = np.array(ivalue_memory)
                old_buffer = np.array(old_memory)
                usteps = math.ceil(len(state_buffer)/self.batch_size)-1

                randomize = np.arange(len(state_buffer))
                np.random.shuffle(randomize)
                state_buffer = state_buffer[randomize]
                advantage_buffer = advantage_buffer[randomize]
                #evalue_buffer = evalue_buffer[randomize]
                ivalue_buffer = ivalue_buffer[randomize]
                old_buffer = old_buffer[randomize]

                preloss = imodel_new.predict(state_buffer)
                imean = np.mean(preloss)
                istd = np.std(preloss)
                imin = np.min(preloss)
                imax = np.max(preloss)
                if self.stats['initialized'] == 1:
                    self.stats['imean'] = imean
                    self.stats['istd']  = istd
                    self.stats['imax'] = imax
                    self.stats['imin'] = imin
                if self.stats['initialized'] > 1:
                    self.stats['imean'] = ((self.horizon - steps) * self.stats['imean'] + steps * imean)/self.horizon
                    self.stats['istd']  = ((self.horizon - steps) * self.stats['istd'] + steps * istd)/self.horizon
                    self.stats['imax'] = ((self.horizon - steps) * self.stats['imax'] + steps * imax)/self.horizon
                    self.stats['imin'] = ((self.horizon - steps) * self.stats['imin'] + steps * imin)/self.horizon
                
                if self.stats['steps'] >self.minimal_steps:
                    print('training policy model...')
                    agenerator = self.batch_generator_simple(state_buffer,
                                                    advantage_buffer,
                                                    old_buffer,
                                                    target='policy')
                    Amodel.fit_generator(generator=agenerator, 
                                                steps_per_epoch=usteps, 
                                                epochs=self.epochs, 
                                                verbose=1)
                # print('training ext value model...')
                # evgenerator = self.batch_generator(state_buffer, advantage_buffer, evalue_buffer, old_buffer, target='value')
                # eVmodel.fit_generator(generator=evgenerator, 
                #                             steps_per_epoch=usteps, 
                #                             epochs=self.epochs, 
                #                             verbose=1)
                print('training ireward new model...')
                # imodel_old.set_weights(imodel_middle.get_weights())
                # imodel_middle.set_weights(imodel_new.get_weights())
                igenerator = self.imodel_generator(state_buffer, self.batch_size)
                imodel_new.fit_generator(generator=igenerator,
                                            steps_per_epoch=usteps, 
                                            epochs=1,
                                            verbose=1)


                # print('training ireward old model...')   
                # imodel_old.fit_generator(generator=igenerator,
                #                             steps_per_epoch=usteps/20, 
                #                             epochs=1, 
                #                             verbose=1)

                self.stats['initialized'] += 1

                print('training int value model...')
                ivgenerator = self.batch_generator_simple(state_buffer, advantage_buffer, old_buffer, target='value', value_buffer=ivalue_buffer)
                iVmodel.fit_generator(generator=ivgenerator, 
                                           steps_per_epoch=usteps, 
                                           epochs=1, 
                                           verbose=1)
                state_buffer = list()
                advantage_buffer = list()
                evalue_buffer = list()
                ivalue_buffer = list()
                old_buffer = list()
                action_memory = list()
                state_memory = list()
                ireward_memory = list()
                ereward_memory = list()
                evalue_memory = list()
                ivalue_memory = list()
                old_memory = list()
                eadvantage_memory = list()
                iadvantage_memory = list()
                advantage_memory = list()

                lock.acquire()
                imodel_new.save_weights(r'D:\sonic_models\imodel_new.h5')
                imodel_old.save_weights(r'D:\sonic_models\imodel_old.h5')
                Amodel.save_weights(r'D:\sonic_models\policy_model.h5')
                #eVmodel.save_weights(r'D:\sonic_models\evalue_model.h5')
                iVmodel.save_weights(r'D:\sonic_models\ivalue_model.h5')

                # l = []
                # for _ in range(math.ceil(self.maxstat/2)):
                #     r=random.randint(math.ceil(len(self.stats['episodes_numbers'])/2),
                #                             len(self.stats['episodes_numbers'])-1)
                #     if r not in l: 
                #         l.append(r)

                for key in self.stats.keys():
                    if isinstance(self.stats[key], list):
                        # if (key not in excluded) and (len(self.stats[key])>self.maxstat):
                        #     new = []
                        #     for r in sorted(l):
                        #         new.append(self.stats[key][r])
                        #     self.stats[key][-len(new):] = new
                        if key not in excluded and len(self.stats[key])>2500:
                            self.stats[key] = self.stats[key][::2]
                        self.stats[key] = list(map(lambda x: float(x), self.stats[key]))
                        
                    elif isinstance(self.stats[key], np.generic):
                        self.stats[key] = float(self.stats[key])
                self.save_stats()

                for elem in excluded:
                    self.stats[elem] = []

                lock.release()
                for i in range(num_workers):
                    new_weights.put(True)
                if self.render:
                    render_que.put(True)
                
                if self.stats['mean100_external_rewards'][-1]>99500:
                    print('agent completed training successfully in '+str(self.stats['episodes_passed'])+' episodes')
                    for i in range(num_workers):
                        last_word = 'process '+str(processes[i].pid)+' terminated'
                        processes[i].terminate()
                        print(last_word)
                    break
    
    def get_stats(self, path=r'D:\sonic_models\results.json'):
        try:
            with open(path, 'r') as file:
                self.stats = json.load(file)
        except:
            print('loading default stats...')
            self.stats['steps'] = 0
            self.stats['steps_list'] = []
            self.stats['episodes_passed'] = 0
            self.stats['initialized'] = 0
            self.stats['episodes_numbers'] = []
            self.stats['emean'] = 1
            self.stats['imean'] = 0
            self.stats['istd'] = 1
            self.stats['imax'] = 1
            self.stats['imin'] = 0
            self.stats['window'] = 10
            self.stats['mean100_external_rewards'] = []
            self.stats['external_rewards'] = []
            self.stats['external_rewards_per_step'] = []
            self.stats['entropy'] = []
            self.stats['mean100_entropy'] = []
            self.stats['some_map'] = []
            self.stats['ireward_map'] = []
            self.stats['entropy_map'] = []
            self.stats['x'] = []
            self.stats['y'] = []
            self.stats['cutted'] = []
            self.stats['action_std'] = []

    def batch_generator_simple(self, state_buffer, advantage_buffer, old_buffer, target, value_buffer=None):
        steps = len(state_buffer)
        while True:
            for ind in range(0, steps-self.batch_size, self.batch_size):
                state_batch = np.array(state_buffer[ind:ind+self.batch_size,:], dtype=np.uint8)
                if target=='policy':
                    dummy_action_batch = np.zeros((self.batch_size,self.action_space))
                    advantage_batch = np.array(advantage_buffer[ind:ind+self.batch_size,:], dtype=np.float32)
                    old_batch = np.array(old_buffer[ind:ind+self.batch_size,:], dtype=np.float32)
                    yield [state_batch, advantage_batch,  old_batch], dummy_action_batch
                elif target=='value':
                    value_batch = np.array(value_buffer[ind:ind+self.batch_size], dtype=np.float32)
                    yield state_batch, value_batch


    def batch_generator_prior(self, state_buffer, advantage_buffer, value_buffer, old_buffer, target):
        steps = len(state_buffer)
        p_small = np.array([(1/(i+1))**0.5 for i in range(steps)])
        p_big = p_small/np.sum(p_small)
        indes = np.argsort(np.max(np.abs(advantage_buffer), axis=-1), axis=0)[::-1]
        allen =math.ceil(steps/self.batch_size) * self.batch_size * self.epochs
        w = 1 / allen / p_big
        w /= np.max(w)
        while True:
            i = 0
            state_batch0 = []
            advantage_batch0 = []
            old_batch0 = []
            value_batch0 = []
            while i < allen:
                ind = np.random.choice(steps, size=None, p=p_big)
                targ = indes[ind]
                state_batch0.append(state_buffer[targ])
                advantage_batch0.append(w[ind] * advantage_buffer[targ,:])
                old_batch0.append(old_buffer[targ,:])
                value_batch0.append(value_buffer[targ,:])
                i += 1
                if len(state_batch0) == self.batch_size:
                    state_batch = np.array(state_batch0, dtype=np.uint8)
                    advantage_batch = np.array(advantage_batch0, dtype=np.float32)
                    old_batch = np.array(old_batch0, dtype=np.float32)
                    value_batch = np.array(value_batch0, dtype=np.float32)
                    state_batch0 = []
                    advantage_batch0 = []
                    old_batch0 = []
                    value_batch0 = []
                    dummy_action_batch = np.zeros((self.batch_size,self.action_space))
                    if target=='policy':
                        yield [state_batch, advantage_batch,  old_batch], dummy_action_batch
                    elif target=='value':
                        yield state_batch, value_batch

    def imodel_generator(self, state_buffer, batch_size):
        steps = len(state_buffer)
        while True:
            for ind in range(0, steps-batch_size, batch_size):
                state_batch = np.array(state_buffer[ind:ind+batch_size,:], dtype=np.uint8)
                dummy_target = np.zeros((batch_size,1))
                yield state_batch, dummy_target
    
    def compute_advantages(self, states, rewards, actions, value_model, gamma, lam, episodic, trace):
        states = self.process_states(states)
        values = value_model.predict(states)
        rewards = np.array(rewards)
        if episodic:
            values = np.vstack((values[:], np.array([[0]])))
        else:
            values = np.vstack((values[:], values[0]))
        G = []
        temp = 0
        for reward in rewards[::-1]:
            temp = reward + gamma * temp
            G.append(temp)
        G.reverse()
        G = np.array(G, dtype=np.float32)
        steps = len(states)
        advantages = np.zeros((steps,self.action_space)).astype(self.adv)
        target_values = np.zeros((steps,1)).astype(self.adv)
        td = np.zeros((steps,1)).astype(self.adv)
        rewards = rewards.reshape((-1,1))
        
        if self.stats['initialized']<50:
            target_values[:,0] = G[:]
        else:
            target_values[:,0] = rewards[:,0] + gamma * values[1:,0]

        
        td[:,0] =  target_values[:,0] - values[:-1,0]
        temp = 0
        gl = gamma * lam
        ind = steps-1
        for elem in td[::-1,0]:
            temp = elem + gl * temp
            advantages[ind, actions[ind]] = temp        
            ind -= 1
        if trace:
            if value_model:
                print('values_mean:', np.mean(values[:self.choosed_length]))
            print('meanG_choosed:', np.mean(G[:self.choosed_length]), 'meanG_all:', np.mean(G))
            print('target_values:', np.mean(target_values[:self.choosed_length]))
            print('advantages_mean:', np.mean(np.abs(advantages[:self.choosed_length])))
            print('advantages_std:', np.std(advantages[:self.choosed_length]))
        return advantages, target_values

    def run_episode(self, Amodel, render=False, record=False, game_id=0, path='.'):
        import retro
        import random
        if record:
            env = retro.make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act2', scenario='contest', record=path)    
        else:
            env = retro.make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act2', scenario='contest')
        env.movie_id = game_id
        cur_mem = deque(maxlen = self.state_len)
        done = False
        delay_limit = self.timedelay
        frame_reward = 0
        state_memory = []
        ereward_memory = []
        action_memory = []
        old_memory = []
        coords = {'x': [], 'y': []}
        for _ in range(self.state_len):
            cur_mem.append(np.zeros((self.height, self.width, 3),dtype=self.default))
        next_state = env.reset()
        while not done:
            next_state = self.resize_state(next_state)
            cur_mem.append(np.copy(next_state))
            state_memory.append(np.copy(cur_mem))
            action_id, cur_policy = self.choose_action(cur_mem, Amodel, render)
            old_memory.append(cur_policy)
            action_memory.append(action_id)
            for _ in range(random.randint(delay_limit-3, delay_limit+3)):
                next_state, reward, done, info = env.step(self.actions[action_id])
                frame_reward += reward
            ereward_memory.append(frame_reward)
            coords['x'].append(info['x'])
            coords['y'].append(info['y'])
            frame_reward = 0
            if render:
                env.render()
        env.render(mode='rgb_array', close=True)
        return state_memory, ereward_memory, action_memory, old_memory, coords
            

    def worker(self, l, data, task_sequence, render_que, use_gpu=True):
        import os
        import time
        if not use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        l.acquire()
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        policy_net = create_policy_net('policy', self.state_len, self.action_space)
        if os.path.isfile(r'D:\sonic_models\policy_model.h5'):
            policy_net.load_weights(r'D:\sonic_models\policy_model.h5')
            policy_loading_time = os.path.getmtime(r'D:\sonic_models\policy_model.h5')
        else:
            policy_loading_time=0
        l.release()

        while True:
            game_id = task_sequence.get()
            render = False
            if render_que.qsize()>0:
                render = render_que.get()
            l.acquire()
            if os.path.isfile(r'D:\sonic_models\policy_model.h5'):
                if policy_loading_time<os.path.getmtime(r'D:\sonic_models\policy_model.h5'):
                    policy_loading_time = os.path.getmtime(r'D:\sonic_models\policy_model.h5')
                    try:
                        policy_net.load_weights(r'D:\sonic_models\policy_model.h5')
                        print('new weights are loaded by process', os.getpid())
                    except:
                        print('new weights are not loaded by process', os.getpid())
            l.release()
            # if game_id in self.special_game_ids:
            #     record_replay=True
            # else:
            record_replay=False
            state_memory, ereward_memory, action_memory, old_memory, infox_memory = self.run_episode(policy_net, render=render, record=record_replay, game_id=game_id, path=r'D:\sonic_models\replays')
            data.put((state_memory, ereward_memory, action_memory, old_memory, infox_memory))

    def choose_action(self, state_col, Amodel, render):
        states = np.reshape(np.array(state_col, dtype=np.uint8), (1, len(state_col), self.height, self.width, 3))
        dummy_rew = np.zeros((1, self.action_space))
        dummy_old = np.zeros((1, self.action_space))
        policy = Amodel.predict([states, dummy_rew, dummy_old])
        policy = policy[-1]
        if render:
            print([round(pol,6) for pol in policy])
        order = np.random.choice(self.action_space, size=None, p=policy)
        return order, policy

    def process_states(self, state_memory):
        states = np.array(state_memory, dtype=np.uint8)
        states.reshape((len(state_memory), self.state_len, self.height, self.width, 3))
        return states.astype(np.uint8)

    def get_actions(self):
        target = []
        buttons = ('B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z')
        #actions = [[], ['LEFT'], ['RIGHT'], ['LEFT','DOWN'], ['RIGHT','DOWN'], ['DOWN'], ['DOWN', 'B'], ['B'], ['LEFT','B'], ['RIGHT','B']]
        actions = [['LEFT'], ['RIGHT'], ['LEFT','B'], ['RIGHT','B'], ['B']]
        for i in range(len(actions)):
            action_list = [0] * len(buttons)
            for button in actions[i]:
                action_list[buttons.index(button)]=1
            target.append(np.array(action_list))
        return np.array(target)    
        
    def resize_state(self, state):
        resized = cv2.resize(state, (self.width, self.height))
        resized = np.array(resized, dtype=np.uint8)
        return resized

    def process_ereward(self, ereward_memory):
        sume = np.sum(ereward_memory)
        #ereward_memory_temp = np.zeros((len(ereward_memory),))
        ereward_memory = np.array(ereward_memory)

        #for i in range(len(ereward_memory)-1):
        #    if np.sum(ereward_memory[:i])>0.02:
        #        ereward_memory_temp[i] = np.sum(ereward_memory[:i])
        #        ereward_memory[:i] *= 0
        #ereward_memory[:] *= 0
        #ereward_memory[:] = ereward_memory_temp[:]
        if sume>=0.99:
            ereward_memory[-1]=1
        else:
            pass #ereward_memory[-1]=10 #-sume/50
        return ereward_memory


                
                
                

            
               



if __name__ == '__main__':
    agent = SonicAgent()
    agent.run_train()
    # result_file = open(r'D:\sonic_models\results.json', 'r')
    # results = json.load(result_file)
    # episode_passed = results['episode_passed']
    # Amodel = create_policy_net('policy', 1)
    # if os.path.isfile(r'D:\sonic_models\policy_model.h5'):
    #     Amodel.load_weights(r'D:\sonic_models\policy_model.h5')
    # agent.run_episode(Amodel, render=True, record=True, game_id=episode_passed, path=r'D:\sonic_models\replays')


