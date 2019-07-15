
import os
import json
import random
import math
import numpy as np
import random
import time
import cv2
from collections import deque
from matplotlib import pyplot as plt
import tensorflow as tf
import queue
from multiprocessing import Queue, Process, Lock

def create_reward_net(path, state_len):
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
    height = 88
    width = 120
    usb = True
    bm = 0.999
    action_space = 10
    #adam1 = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-5, decay=0.0, amsgrad=False)
    #adam2 = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0, clipnorm=10., amsgrad=False)
    adam1 = SGD(lr=1e-4, momentum=0.0, decay=0.0, nesterov=False, clipvalue=50.0)
    #adam2 = SGD(lr=1e-2, momentum=0.0, decay=0.0, nesterov=False)
    def reward_loss(y_true, y_pred):
        return K.mean(K.abs(y_pred-y_true))
    def last_image(tensor):
        return tensor[:,-1,:]
    def ireward(tensor):
        return K.mean(0.25 * K.mean(K.square(tensor[0] - tensor[1]),axis=-1))

    main_input = Input(shape=(state_len, height, width, 3))
    float_input = K.cast(main_input, dtype='float32')
    float_input = Lambda(lambda input: input/127.5-1)(float_input)
#    input_last = Lambda(last_image)(float_input)
    xs = TimeDistributed(Conv2D(40, (8,8), strides=(4,4), padding='same', use_bias=usb))(float_input)
    xs = TimeDistributed(Activation('tanh'))(xs)
    xs = TimeDistributed(MaxPooling2D((4, 4), padding='same'))(xs)
    print(xs.shape)
    xs = TimeDistributed(Conv2D(80, (4,4), strides=(2,2), padding='same', use_bias=usb))(xs)
    xs = TimeDistributed(Activation('tanh'))(xs)
    xs = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(xs)
    print(xs.shape)
    xs = TimeDistributed(Conv2D(80, (4,4), strides=(1,1), padding='same', use_bias=usb))(xs)
    xs = TimeDistributed(Activation('tanh'))(xs)
    #square_size = K.int_shape(xs)
    #print('rew_sq:', square_size)
    xs = TimeDistributed(Flatten())(xs)
    #flatten_size = K.int_shape(xs)
    #print('rew_fl:', flatten_size)
    #xs = TimeDistributed(Dense(512))(xs)
    #xs = TimeDistributed(Dense(flatten_size[2]))(xs)
    #xs = Reshape(square_size[1:])(xs)
    #xs = TimeDistributed(Conv2D(64, (1,1), padding='same'))(xs)
    #xs = TimeDistributed(Activation('tanh'))(xs)
    #print(xs.shape)
    #xs = TimeDistributed(Conv2D(64, (4,4), padding='same'))(xs)
    #xs = TimeDistributed(Activation('tanh'))(xs)
    #xs = TimeDistributed(UpSampling2D((2, 2)))(xs)
    #print(xs.shape)
    #xs = TimeDistributed(Conv2DTranspose(32, (4,4), padding='same'))(xs)
    #xs = TimeDistributed(Activation('tanh'))(xs)
    #xs = TimeDistributed(UpSampling2D((4, 4)))(xs)
    #print(xs.shape)
    #xs = TimeDistributed(Conv2D(3, (8,8), padding='same'))(xs)
    #print(xs.shape)
    #output_last = Lambda(last_image)(xs)
    #intrinsic_reward = Lambda(ireward)([input_last, output_last])
    #intrinsic_reward = Lambda(ireward)([stochastic_features, predicted_features])
    xs= Lambda(last_image)(xs)
    feature_dimension = K.int_shape(xs)
    print('feature dimension:', feature_dimension)
    xs = Dense(512, use_bias=usb)(xs)
    xs = Activation('tanh')(xs)
    xs = Dense(512, use_bias=usb)(xs)
    stochastic_features = Activation('tanh')(xs)
    stochastic_part = Model(inputs=main_input, outputs=stochastic_features)
    for layer in stochastic_part.layers:
        layer.trainable = False
    xt = TimeDistributed(Conv2D(40, (8,8), strides=(4,4), padding='same', use_bias=usb))(float_input)
    xt = TimeDistributed(Activation('tanh'))(xt)
    xt = TimeDistributed(Conv2D(80, (4,4), strides=(2,2), padding='same', use_bias=usb))(xt)
    xt = TimeDistributed(Activation('tanh'))(xt)
    xt = TimeDistributed(Conv2D(80, (4,4), strides=(1,1), padding='same', use_bias=usb))(xt)
    xt = TimeDistributed(Activation('tanh'))(xt)
    xt = TimeDistributed(Flatten())(xt)
    xt= Lambda(last_image)(xt)
    feature_dimension = K.int_shape(xt)[1]
    print('feature dimension:', feature_dimension)
    xt = Dense(512, use_bias=usb)(xt)
    xt = Activation('tanh')(xt)
    xt = Dense(512, use_bias=usb)(xt)
    predicted_features = Activation('tanh')(xt)
    intrinsic_reward = Lambda(ireward)([stochastic_features, predicted_features])

    model = Model(inputs=main_input,outputs=intrinsic_reward)
    model.compile(optimizer=adam1, 
            loss=reward_loss)
    exists = os.path.isfile(path)
    if exists:
        model.load_weights(path)
        print('weights loaded from '+path)
    #model.summary()
    return model

def create_policy_net(path, state_len):
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
    from keras.regularizers import l2
    from keras import losses
    from keras import regularizers
    from keras.models import Model
    from keras import backend as K
    from keras.losses import mean_squared_error
    from keras.losses import categorical_crossentropy
    from keras import regularizers
    import math
    beta = 0.00000
    height = 88
    width = 120
    usb = True
    crange = 0.1
    action_space = 10
    adam1 = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-5, decay=0.0, amsgrad=False)
    #adam2 = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0, clipnorm=10., amsgrad=False)
    #adam1 = SGD(lr=1e-4, momentum=0.0, decay=0.0, nesterov=False, clipvalue=50.0)
    #adam2 = SGD(lr=1e-2, momentum=0.0, decay=0.0, nesterov=False)
    def sample_loss(reward_input, old_input):
        def custom_loss(y_true, y_pred):
            entropy = -K.sum(y_pred * K.log(y_pred), axis=-1) / np.log(action_space)
            ratio = y_pred/(old_input+1e-3)
            pg_loss1 = -reward_input * ratio
            pg_loss2 = -reward_input * K.clip(ratio, 1.0 - crange, 1.0 + crange)
            pg_loss = K.maximum(pg_loss1,pg_loss2)
            return K.sum(pg_loss,axis=-1) - beta * entropy
            #return -K.mean(K.sum(reward_input * K.log(y_pred+1e-3), axis=1), axis=0)# + K.mean(K.sum(0.05 * K.square(y_pred/(old_input)-1),axis=1),axis=0)
        return custom_loss

    def critic_loss(y_true, y_pred):
        return K.mean(0.5 * K.square(y_true - y_pred), axis=-1)
    def last_image(tensor):
        return tensor[:,-1,:]
    
    reward_input = Input(shape=(action_space,))
    old_input = Input(shape=(action_space,))
    main_input = Input(shape=(state_len, height, width, 3))
    float_input = K.cast(main_input, dtype='float32')
    float_input = Lambda(lambda input: input/127.5-1)(main_input)
    xm = TimeDistributed(Conv2D(40, (8,8), strides=(4,4), padding='same', use_bias=usb))(float_input)
    xm = TimeDistributed(Activation('tanh'))(xm)
    xm = TimeDistributed(Conv2D(80, (4,4), strides=(2,2), padding='same', use_bias=usb))(xm)
    xm = TimeDistributed(Activation('tanh'))(xm)
    xm = TimeDistributed(Conv2D(80, (4,4), strides=(1,1), padding='same', use_bias=usb))(xm)
    xm = TimeDistributed(Activation('tanh'))(xm)
    xm = TimeDistributed(Flatten())(xm)
    xm1= Lambda(last_image)(xm)
    feature_dimension = int(xm1.shape[1])
    print('feature shape is: ', feature_dimension)
    xm = Dense(512, use_bias=usb)(xm1)
    xm = Activation('tanh')(xm)
    xm = Dense(256, use_bias=usb)(xm)
    xm = Activation('tanh')(xm)
    #xm = CuDNNLSTM(10, return_sequences=False )(xm)
    main_output = Dense(action_space, activation='softmax',  use_bias=usb, name='main_output')(xm)

    xc = Dense(512, use_bias=usb)(xm1)
    xc = Activation('tanh')(xc)
    xc = Dense(256, use_bias=usb)(xc)
    xc = Activation('tanh')(xc)
    #xc = CuDNNLSTM(10, return_sequences=False )(xc)
    critic_output = Dense(1, activation='tanh', use_bias=usb, name='critic_output')(xc)
    model = Model(inputs=[main_input,reward_input,old_input],outputs=[main_output, critic_output])

    if path==r'E:\sonic_models\policy_model.h5':
        weights={'main_output': 1, 'critic_output': 0}
    elif path==r'E:\sonic_models\evalue_model.h5' or path==r'E:\sonic_models\ivalue_model.h5':
        weights={'main_output': 0, 'critic_output': 1}
    model.compile(optimizer=adam1, 
            loss={'main_output': sample_loss(reward_input,old_input), 'critic_output': critic_loss}, 
            loss_weights=weights)
    exists = os.path.isfile(path)
    if exists:
        model.load_weights(path)
        print('weights loaded from '+path)
    #model.summary()
    return model
    

class SonicAgent():
    def __init__(self, episodes_step=10, max_env_steps=None, state_len=1, gamma=0.999, batch_size=128, workers=8, render=False):
        self.ls = 40
        self.workers = workers
        self.multisteps = 10
        self.iw = 0.1
        self.ew = 1
        self.epochs = 1
        self.action_space = 10
        self.state_len = state_len
        self.epsilon_max = 1
        self.batch_size = batch_size
        self.height = 88
        self.width = 120
        self.gamma = gamma
        self.egamma = gamma
        self.igamma = 0.99
        self.episodes = episodes_step
        self.beta = 0.1
        self.timedelay = 15
        self.V = []
        self.old = []
        self.tau = 50000
        self.default = np.float32
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
        self.imean = 0
        self.istd = 1
        self.imax = 1
        self.imin = 0
        self.meanent = 1
        self.render = render

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



    def choose_action(self, state_col, Amodel, render):
        states = np.reshape(np.array(state_col, dtype=np.uint8), (1, len(state_col), self.height, self.width, 3))
        zero_rew = np.zeros((1, self.action_space))
        zero_old = np.zeros((1, self.action_space))
        policy, temp_value = Amodel.predict([states, zero_rew, zero_old])
        policy = policy[-1]
        if render:
            print([round(policy[i],6) for i in range(len(policy))])
        if random.random() < 0:#0.05 + 0.95*math.exp(-self.training_steps / self.tau):
            order = random.randint(0,9)
            order = self.convert_order(order)
        else:
            order = self.convert_order(np.random.choice(self.action_space, size=None, p=policy))
        self.steps += 1
        self.steps2 += 1
        return order, policy
        
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

    def process_ereward(self, ereward_memory):
        ereward_memory_temp = np.zeros((len(ereward_memory),))
        ereward_memory = np.array(ereward_memory)
        for i in range(len(ereward_memory)-1):
            if np.sum(ereward_memory[:i])>0.1:
                ereward_memory_temp[i+1] = np.sum(ereward_memory[:i])
                ereward_memory[:i] *= 0
        ereward_memory[:] *= 0
        ereward_memory[:] = ereward_memory_temp[:]
        return np.array(ereward_memory_temp)

    def get_entropy_reward(self, old_memory):
        old = np.array(old_memory)
        entropy = -np.sum(old * np.log(old), axis=-1) / (-np.log(1/self.action_space))
        meanent = np.mean(entropy)
        self.meanent = 0.995 * self.meanent + 0.005 * meanent
        entropy -= self.meanent
        entropy = 0.1 * np.where(entropy>0, entropy, 0)
        print('sum ent:', np.sum(entropy), 'entropy_mean:', meanent)
        return entropy / (36000 / self.timedelay)
    
    def process_states(self, state_memory):
        return np.array(state_memory, dtype=np.uint8).reshape((len(state_memory), self.state_len, self.height, self.width, 3)).astype(np.uint8)

    def get_ireward(self, state_memory, imodel, only_last=False):
        states = self.process_states(state_memory)
        irewards = imodel.predict(states)[1:]
        imean = np.mean(irewards)
        istd = np.std(irewards)
        imin = np.min(irewards)
        imax = np.max(irewards)
        m = 0.9 + 0.195 * (1-math.exp(-self.episode_passed/2000))
        if self.episode_passed > 100:
            self.imean = m * self.imean + (1-m) * imean
            self.istd  = m * self.istd + (1-m) * istd
            self.imax = m * self.imax + (1-m) * imax
            self.imin = m * self.imin + (1-m) * imin
        elif self.episode_passed == 100:
            self.imean = imean
            self.istd = istd
            self.imax = imax
            self.imin = imin
        irewards -= self.imean
        irewards /= self.imax
        if only_last:
            irewards[:-only_last] = 0
            irewards /= only_last
        else:
            irewards /= 36000 / self.timedelay
        irewards = np.where(irewards>0, irewards, 0)
        print('ireward_sum:', np.sum(irewards))
        return irewards

    def compute_advantages(self, state_memory, reward_memory, action_memory, value_model, gamma, weight):
        states = self.process_states(state_memory)
        zero_rew = np.zeros((len(state_memory), self.action_space))
        zero_old = np.zeros((len(state_memory), self.action_space))
        policy_temp, values = value_model.predict([states, zero_rew, zero_old])
        V_current = values[:-1,0]
        V_next = values[1:,0]
        print('vmean:', np.mean(V_current))
        G = []
        temp = 0
        for reward in reward_memory[::-1]:
            temp = weight * reward + temp * gamma
            G.append(temp)
        G.reverse()
        G = np.array(G, dtype=np.float32)
        print('maxG:', np.max(G))
        steps = len(V_current)
        actions = np.zeros((steps,self.action_space))
        advantages = np.zeros((steps,self.action_space)).astype(self.adv)
        target_values = np.zeros((steps,1)).astype(self.adv)
        td = np.zeros((steps,1)).astype(self.adv)
        multistep = self.multisteps
        for ind in range(steps):
            if ind+multistep < steps:
                temp = 0
                for reward in reward_memory[ind+multistep:ind:-1]:
                    temp = weight * reward + temp * gamma
                td[ind] = temp + gamma**(multistep) * V_next[ind+multistep-1] - V_current[ind]
                target_values[ind,0] = temp + gamma**(multistep) * V_next[ind+multistep-1]
                advantages[ind, action_memory[ind]] = td[ind]
            else:
                td[ind] = G[ind] - V_current[ind]
                target_values[ind,0] = G[ind]
                advantages[ind, action_memory[ind]] = td[ind]
        print('values:', np.mean(target_values))
        return advantages, target_values

    def run_train(self):
        lock = Lock()
        lock.acquire()
        exists = os.path.isfile(r'E:\sonic_models\results.json')
        if exists:
            result_file = open(r'E:\sonic_models\results.json', 'r')
            try:
                results = json.load(result_file)
                self.imean = results['imean']
                self.imin = results['imin']
                self.imax = results['imax']
                self.istd = results['istd']
                self.meanent = results['meanent']
                self.episode_passed = results['episode_passed']
                self.mean_score = results['mean_score']
            except:
                results = {}
                self.episode_passed = 0
            result_file.close()
        else:
            results = {}
            self.episode_passed = 0
        
        num_episodes = self.episodes
        num_workers = self.workers
        self.scores= []
        path = None
        data = Queue()
        task_sequence = Queue()
        new_weights = Queue()
        render_que = Queue()
        render_que.put(True)
        
        from keras import backend as K
        K.set_floatx('float32')
        gpu_options = tf.GPUOptions(allow_growth=True)
        K.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
        Amodel = create_policy_net(r'E:\sonic_models\policy_model.h5', self.state_len)
        Amodel.summary()
        eVmodel = create_policy_net(r'E:\sonic_models\evalue_model.h5', self.state_len)
        iVmodel = create_policy_net(r'E:\sonic_models\ivalue_model.h5', self.state_len)
        imodel = create_reward_net(r'E:\sonic_models\imodel.h5', self.state_len)
        imodel.summary()
        lock.release()

        for i in range(num_episodes):
            task_sequence.put(i)
        processes = [0] * num_workers
        process_statuses = [None] * num_workers
        for i in range(num_workers):
            processes[i] = Process(target=self.worker, args=(lock, path, data, task_sequence, render_que, new_weights))
            processes[i].start()
        print(processes)
        k = 0
        state_buffer = list()
        advantage_buffer = list()
        evalue_buffer = list()
        ivalue_buffer = list()
        old_buffer = list()
        while True:
            if data.qsize() >= 1:
                self.episode_passed += 1
                k += 1
                (state_memory, ereward_memory, action_memory, old_memory, infox_memory) = data.get()
                episode_reward = sum(ereward_memory)
                self.scores.append(episode_reward)
                if self.episode_passed == 1:
                    self.mean_score = episode_reward
                else:
                    self.mean_score = self.mean_score*0.99 + 0.01 * episode_reward
                print('episode: ', self.episode_passed, 'episode_reward: ', episode_reward, 'mean_reward: ', self.mean_score)
                eadvantage_memory, etarget_values = self.compute_advantages(state_memory, ereward_memory, action_memory, eVmodel, self.egamma, self.ew)
                irewards = self.get_ireward(state_memory, imodel, only_last=False)
                entropy_reward = self.get_entropy_reward(old_memory)
                iadvantage_memory, itarget_values = self.compute_advantages(state_memory, irewards, action_memory, iVmodel, self.igamma, self.iw)
                
                print('----------------------------------------------------------------------------------------------------')
                state_buffer.extend(state_memory[:-1])
                advantage_buffer.extend(eadvantage_memory+iadvantage_memory)
                evalue_buffer.extend(etarget_values)
                ivalue_buffer.extend(itarget_values)
                old_buffer.extend(old_memory)
            if  k == num_episodes:
                steps = math.ceil(len(evalue_buffer)/self.batch_size)-1
                if steps>2:
                    state_buffer = np.array(state_buffer, dtype=np.uint8)
                    advantage_buffer = np.array(advantage_buffer)
                    evalue_buffer = np.array(evalue_buffer)
                    ivalue_buffer = np.array(ivalue_buffer)
                    old_buffer = np.array(old_buffer)
                    agenerator = self.batch_generator(state_buffer, advantage_buffer, evalue_buffer, old_buffer, prioretization=False)
                    Amodel.fit_generator(generator=agenerator, 
                                                steps_per_epoch=steps, 
                                                epochs=self.epochs, 
                                                verbose=1)
                    evgenerator = self.batch_generator(state_buffer, advantage_buffer, evalue_buffer, old_buffer, prioretization=False)
                    eVmodel.fit_generator(generator=evgenerator, 
                                                steps_per_epoch=steps, 
                                                epochs=self.epochs, 
                                                verbose=1)
                    ivgenerator = self.batch_generator(state_buffer, advantage_buffer, ivalue_buffer, old_buffer, prioretization=False)
                    iVmodel.fit_generator(generator=ivgenerator, 
                                                steps_per_epoch=steps, 
                                                epochs=self.epochs, 
                                                verbose=1)
                    igenerator = self.ireward_batch_generator(state_buffer, prioretization=False)
                    imodel.fit_generator(generator=igenerator,
                                                steps_per_epoch=math.ceil(steps/100), 
                                                epochs=1, 
                                                verbose=1)
                    state_buffer = list()
                    advantage_buffer = list()
                    evalue_buffer = list()
                    ivalue_buffer = list()
                    old_buffer = list()
                    imodel.save_weights(r'E:\sonic_models\imodel.h5')
                    Amodel.save_weights(r'E:\sonic_models\policy_model.h5')
                    eVmodel.save_weights(r'E:\sonic_models\evalue_model.h5')
                    iVmodel.save_weights(r'E:\sonic_models\ivalue_model.h5')
                    result_file = open(r'E:\sonic_models\results.json', 'w+')
                    results['imin'] = self.imin 
                    results['imean'] = self.imean
                    results['imax'] = self.imax 
                    results['istd'] = self.istd
                    results['meanent'] = self.meanent
                    results['episode_passed'] = self.episode_passed
                    results['mean_score'] = self.mean_score
                    json.dump(results, result_file)
                    result_file.close()
                    for i in range(num_workers):
                        new_weights.put(True)
                if steps > 100 and num_episodes>num_workers:
                    num_episodes -= 1
                else:
                    num_episodes += 1
                k = 0
                if self.render:
                    render_que.put(True)

                for i in range(num_episodes):
                    task_sequence.put(i)

        for i in range(num_workers):
            processes[i].join()
        print('all processes performed')

    def batch_generator(self, state_buffer, advantage_buffer, value_buffer, old_buffer, prioretization=False):
        while True:
            if prioretization:
                steps = len(self.value_buffer)
                p_small = np.array([(1/(i+1))**0.5 for i in range(steps)])
                p_big = p_small/np.sum(p_small)
                indes = np.argsort(np.max(np.abs(advantage_buffer), axis=-1), axis=0)[::-1]
                allen =math.ceil(steps/self.batch_size) * self.batch_size * self.epochs
                w = 1 / allen / p_big
                w /= np.max(w)
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
                        zero_action_batch = np.zeros((self.batch_size,self.action_space))
                        yield [state_batch, advantage_batch,  old_batch], [zero_action_batch,  value_batch]
            else:
                randomize = np.arange(len(state_buffer))
                np.random.shuffle(randomize)
                state_buffer = state_buffer[randomize]
                advantage_buffer = advantage_buffer[randomize]
                value_buffer = value_buffer[randomize]
                old_buffer = old_buffer[randomize]
                for ind in range(0, len(value_buffer)-self.batch_size, self.batch_size):
                    state_batch = np.array(state_buffer[ind:ind+self.batch_size,:], dtype=np.uint8)
                    advantage_batch = np.array(advantage_buffer[ind:ind+self.batch_size,:], dtype=np.float32)
                    old_batch = np.array(old_buffer[ind:ind+self.batch_size,:], dtype=np.float32)
                    value_batch = np.array(value_buffer[ind:ind+self.batch_size], dtype=np.float32)
                    zero_action_batch = np.zeros((self.batch_size,self.action_space))
                    yield [state_batch, advantage_batch,  old_batch], [zero_action_batch,  value_batch]

    def ireward_batch_generator(self, state_buffer, prioretization=False):
        while True:
            if prioretization:
                steps = len(state_buffer)
                p_small = np.array([(1/(i+1))**1 for i in range(steps)])
                p_big = p_small/np.sum(p_small)
                indes = np.argsort(irewards, axis=0)[::-1]
                allen =math.ceil(steps/self.batch_size) * self.batch_size * self.epochs
                i = 0
                state_batch0 = []
                while i < allen:
                    ind = np.random.choice(steps, size=None, p=p_big)
                    targ = indes[ind]
                    state_batch0.append(state_buffer[targ])
                    i += 1
                    if len(state_batch0) == self.batch_size:
                        state_batch = np.array(state_batch0, dtype=np.uint8)
                        state_batch0 = []
                        zero_ireward_batch = np.zeros((self.batch_size,1))
                        yield state_batch, zero_ireward_batch
            else:
                randomize = np.arange(len(state_buffer))
                np.random.shuffle(randomize)
                state_buffer = state_buffer[randomize]
                for ind in range(0, len(state_buffer)-self.batch_size, self.batch_size):
                    state_batch = np.array(state_buffer[ind:ind+self.batch_size,:], dtype=np.uint8)
                    zero_ireward_batch = np.zeros((self.batch_size,1))
                    yield state_batch, zero_ireward_batch

    def run_episode(self, Amodel, render=False):
        import retro
        from retrowrapper import RetroWrapper
        env = RetroWrapper(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1', scenario='contest')
        cur_mem = deque(maxlen = self.state_len)
        done = False
        self.substep = 0
        episode_reward = 0
        control_count = 0
        frame_reward = 0
        state_memory = []
        ereward_memory = []
        action_memory = []
        old_memory = []
        infox = []
        for j in range(self.state_len):
            cur_mem.append(np.zeros((self.height, self.width, 3),dtype=self.default))
        state = self.wrapped_reset(env)
        cur_mem.append(np.copy(state))
        state_memory.append(np.copy(cur_mem))
        (action, adesc, adescn), cur_policy = self.choose_action(cur_mem, Amodel, render)
        old_memory.append(cur_policy)
        while True:
            if (control_count == self.timedelay) or done:
                state_memory.append(np.copy(cur_mem))
                action_memory.append(adescn)
                ereward_memory.append(frame_reward)
                infox.append(info['x'])
                if done:
                    return state_memory, ereward_memory, action_memory, old_memory, infox

                if random.random() < 0:
                    (temp1, temp2, temp3), cur_policy = self.choose_action(cur_mem, Amodel, render)  
                else:    
                    (action, adesc, adescn), cur_policy = self.choose_action(cur_mem, Amodel, render)
                old_memory.append(cur_policy)
                next_state, reward, done, info = self.wrapped_step(action, env)
                if render:
                    #print('info:', info)
                    env.render()
                frame_reward = 0
                control_count = 0
            else:
                action, adesc, adescn = self.convert_order(adesc)
                next_state, reward, done, info = self.wrapped_step(action, env)
            if self.substep == 3:
                cur_mem.append(np.copy(next_state))
                self.substep = 0
            self.substep += 1
            control_count += 1
            frame_reward += reward

    def worker(self, l, path, data, task_sequence, render_que, new_weights, use_gpu=True):
        import os
        if not use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        import sys
        from keras import backend as K
        K.set_floatx('float32')
        K.set_epsilon(1e-9)
        gpu_options = tf.GPUOptions(allow_growth=True)
        K.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
        policy_net = create_policy_net('E:\sonic_models\policy_model.h5', 1)
        
        while True:
            l.acquire()
            item = task_sequence.get()
            render = False
            if render_que.qsize()>0:
                render = render_que.get()
            if new_weights.qsize()>0:
                new_weights.get()
                try:
                    policy_net.load_weights(r'E:\sonic_models\policy_model.h5')
                    print('new weights are loaded by process', os.getpid())
                except:
                    print('new weights are not loaded by process', os.getpid())
            l.release()
            state_memory, ereward_memory, action_memory, old_memory, infox_memory = self.run_episode(policy_net, render)
            data.put((state_memory, ereward_memory, action_memory, old_memory, infox_memory))
                
                
                

            
               



if __name__ == '__main__':
    agent = SonicAgent()
    agent.run_train()



