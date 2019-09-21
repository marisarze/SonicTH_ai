
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
    crange = 0.001
    action_space = 10
    #adam1 = Adam(lr=2e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0, amsgrad=False)
    #adam2 = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0, clipnorm=10., amsgrad=False)
    adam1 = SGD(lr=1e-4, momentum=0.999, decay=0.0, nesterov=True, clipvalue=50.0)
    #adam2 = SGD(lr=1e-2, momentum=0.0, decay=0.0, nesterov=False)
    def reward_loss(old_loss):
        def custom_loss(y_true, y_pred):
            return K.square(y_pred)
        return custom_loss

    def last_image(tensor):
        return tensor[:,-1,:]
    def ireward(tensor):
        return 0.5 * K.mean(K.abs(tensor[0] - tensor[1]),axis=-1)
    old_loss = Input(shape=(1,))
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

    model = Model(inputs=[main_input, old_loss], outputs=intrinsic_reward)
    model.compile(optimizer=adam1, 
            loss=reward_loss(old_loss))
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
    beta = 0.00
    height = 88
    width = 120
    usb = True
    crange = 0.2
    action_space = 10
    #adam1 = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-5, decay=0.0, amsgrad=False)
    #adam2 = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0, clipnorm=10., amsgrad=False)
    #adam1 = SGD(lr=1e-4, momentum=0.0, decay=0.0, nesterov=False, clipvalue=50.0)
    #adam2 = SGD(lr=1e-2, momentum=0.0, decay=0.0, nesterov=False)
    def sample_loss(reward_input, old_input):
        def custom_loss(y_true, y_pred):
            entropy = -K.sum(y_pred * K.log(y_pred), axis=-1) / np.log(action_space)
            entropy0 = -K.sum(old_input * K.log(old_input), axis=-1) / np.log(action_space)
            #reward_input1 = reward_input + K.abs(K.sum(reward_input, axis=-1)) * 0.01 * entropy
            ratio = y_pred/(old_input+1e-3)
            pg_loss1 = -reward_input * ratio
            pg_loss2 =-reward_input * K.clip(ratio, 1.0 - crange, 1.0 + crange)
            pg_loss = K.maximum(pg_loss1,pg_loss2)
            return K.sum(pg_loss,axis=-1) #- beta * K.abs(K.sum(reward_input,axis=-1)) * entropy#- beta * (1-entropy0) * entropy
            #return -K.sum(reward_input * K.log(y_pred+1e-3), axis=-1) # + K.mean(K.sum(0.05 * K.square(y_pred/(old_input)-1),axis=1),axis=0)
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
    xm1 = TimeDistributed(Flatten())(xm)
    #xm1= Lambda(last_image)(xm)
    feature_dimension = int(xm1.shape[1])
    print('feature shape is: ', feature_dimension)
    xm = TimeDistributed(Dense(512, use_bias=usb))(xm1)
    xm = TimeDistributed(Activation('tanh'))(xm)

    #xm = Dense(256, use_bias=usb)(xm)
    #xm = Activation('tanh')(xm)
    xm = CuDNNLSTM(10, return_sequences=False )(xm)
    main_output = Dense(action_space, activation='softmax',  use_bias=usb, name='main_output')(xm)

    xc = TimeDistributed(Dense(512, use_bias=usb))(xm1)
    xc = TimeDistributed(Activation('tanh'))(xc)
    #xc = Dense(256, use_bias=usb)(xc)
    #xc = Activation('tanh')(xc)
    xc = CuDNNLSTM(10, return_sequences=False )(xc)
    critic_output = Dense(1, activation='linear', use_bias=usb, name='critic_output')(xc)
    model = Model(inputs=[main_input,reward_input,old_input],outputs=[main_output, critic_output])

    if path==r'E:\sonic_models\policy_model.h5':
        weights={'main_output': 1, 'critic_output': 0}
        adam1 = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0, amsgrad=False)
        #adam1 = SGD(lr=1e-4, momentum=0.999, decay=0.0, nesterov=False, clipvalue=50.0)
    elif path==r'E:\sonic_models\evalue_model.h5' or path==r'E:\sonic_models\ivalue_model.h5':
        weights={'main_output': 0, 'critic_output': 1}
        adam1 = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0, amsgrad=False)
        #adam1 = SGD(lr=1e-4, momentum=0.999, decay=0.0, nesterov=False, clipvalue=50.0)
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
    def __init__(self, episodes_step=8, max_env_steps=None, state_len=3, gamma=0.99, batch_size=256, workers=6, render=False):
        self.critical = 1
        self.csteps = 8 * 2400 / batch_size
        self.ls = 50
        self.workers = workers
        self.multisteps = 3600
        self.iw = 0.0
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
        self.igamma = 0.999
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
        self.imean = 1
        self.imean2 = 1
        self.istd = 1
        self.istd2 = 1
        self.imax = 1
        self.imin = 0
        self.meanent = 1
        self.maxent = 1
        self.render = render
        self.delim = 200
        self.tempmax = 0
        self.counts = 0
        self.norm = 1
        self.mean_score = []
        self.scores = []
        self.rps = []
        self.rpsm = []
        self.rx = [0] * 1000
        self.arx = [0] * 1000
        self.enx = [0] * 1000
        self.perc = [0] * 1000

    def get_entropy_reward(self, old_memory):
        old = np.array(old_memory)
        entropy = -np.sum(old * np.log(old), axis=-1) / (-np.log(1/self.action_space))
        meanent = np.mean(entropy)
        maxent = np.max(entropy)
        print('mean_ent:', meanent)
        #entropy /= self.maxent
        ##entropy[:-1] = entropy[1:]
        ##entropy[-1] = 0
        #entropy = np.where(entropy>1,1,entropy)
        steps = len(entropy)
        m = 1500* 100
        if self.episode_passed == 1:
            self.meanent = meanent
        else:
            self.meanent = ((m - steps) * self.meanent + steps * meanent)/m #0.995 * self.meanent + 0.005 * meanent
        if maxent >= self.tempmax:
            self.tempmax = float(maxent)
        self.counts += 1
        if self.counts>5:
            self.maxent = ((m - steps) * self.maxent + steps * maxent)/m
            self.tempmax = 0
            self.counts = 0
        #for i in range(1,steps-1):
        #    entropy[i] = np.mean(entropy[i-1:i+1])
        return entropy
    
    def get_imodel_result(self, state_memory, imodel):
        steps = len(state_memory)
        states = self.process_states(state_memory)
        zero_old = np.zeros((len(state_memory,)))
        imodel_loss = imodel.predict([states, zero_old])
        temp = imodel_loss[:]
        #for i in range(10):
        #    for i in range(1,steps-1):
        #        temp[i]=np.mean(temp[i-1:])
        #    temp[-1]=np.mean(temp[-2:])
        temp = np.array(list(map(lambda x:float(Decimal(float(x-self.imean))/Decimal(float(self.istd))), temp)))
        
        #temp = np.clip(temp, -10.0, 10.0)
        print('tempmax', np.max(temp))
        print('tempmin', np.min(temp))
        istd2 = np.std(temp)
        imean2 = np.mean(temp)
        #temp = np.array(list(map(lambda x:float(Decimal(1.6)**Decimal(x)), temp)))
        
        #temp /= 10*self.istd2
        #np.clip(temp,-1,1)
        irewards = temp[1:] #- self.imean/self.istd #temp[:-1]
        irewards[:-25] = 0
        irewards[-25:] /= 25
        imean = np.mean(imodel_loss)
        istd = np.std(imodel_loss)
        imin = np.min(imodel_loss)
        imax = np.max(imodel_loss)
        m = 2400* 100#(5 + 200 * (1-math.exp(-self.episode_passed/40)))
        if self.episode_passed > self.critical:
            self.imean = ((m - steps) * self.imean + steps * imean)/m
            self.imean2 = ((m - steps) * self.imean2 + steps * imean2)/m
            self.istd  = ((m - steps) * self.istd + steps * istd)/m
            self.istd2 = ((m - steps) * self.istd2 + steps * istd2)/m
            self.imax = ((m - steps) * self.imax + steps * imax)/(m)
            self.imin = ((m - steps) * self.imin + steps * imin)/(m)
        #elif self.episode_passed == self.critical:
        #    self.imean = imean
        #    self.imean2 = imean2
        #    self.istd = istd
        #    self.istd2 = istd2
        #    self.imax = imax
        #    self.imin = imin
        return imodel_loss, irewards

    def get_art_reward(self,entropy_reward, imodel_loss):
        multi = np.array(list(map(lambda x:1e60**x, imodel_loss-imodel_loss[0])))
        multi /= 10*self.istd
        artificial_reward = multi[1:] - multi[:-1]
        return artificial_reward

    def run_train(self):
        lock = Lock()
        lock.acquire()
        #exists_res = os.path.isfile(r'E:\sonic_models\results.json')
        #exists_sco = os.path.isfile(r'E:\sonic_models\scores.json')

        results = {}
        sco_results = {}
        self.episode_passed = 0
        num_episodes = self.episodes
        try:
            result_file = open(r'E:\sonic_models\results.json', 'r')
            score_file = open(r'E:\sonic_models\scores.json', 'r')
            results = json.load(result_file)
            sco_results = json.load(score_file)
            self.imean = results['imean']
            self.imean2 = results['imean2']
            self.imin = results['imin']
            self.imax = results['imax']
            self.istd = results['istd']
            self.istd2 = results['istd2'] 
            self.meanent = results['meanent']
            self.maxent = results['maxent']
            self.episode_passed = results['episode_passed']
            self.mean_score = sco_results['mean_score']
            num_episodes = results['num_episodes']
            self.counts = results['counts']
            self.tempmax = results['tempmax']
            self.scores = sco_results['scores']
            self.rps = sco_results['rps']
            self.rpsm = sco_results['rpsm']
            self.rx = sco_results['rx']
            self.arx = sco_results['arx']
            self.enx = sco_results['enx']
            self.perc = sco_results['perc']
            score_file.close()    
            result_file.close()
        except:
            results = {}
            sco_results = {}
            self.episode_passed = 0
        
        num_workers = self.workers
        path = None
        data = Queue()
        task_sequence = Queue()
        new_weights = Queue()
        render_que = Queue()
        render_que.put(True)
        
        from keras import backend as K
        K.set_floatx('float32')
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        K.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))
        Amodel = create_policy_net(r'E:\sonic_models\policy_model.h5', self.state_len)
        Amodel.summary()
        eVmodel = create_policy_net(r'E:\sonic_models\evalue_model.h5', self.state_len)
        iVmodel = create_policy_net(r'E:\sonic_models\ivalue_model.h5', self.state_len)
        imodel = create_reward_net(r'E:\sonic_models\imodel.h5', self.state_len)
        imodel.summary()
        lock.release()


        processes = [0] * num_workers
        process_statuses = [None] * num_workers
        for i in range(num_workers):
            processes[i] = Process(target=self.worker, args=(lock, path, data, task_sequence, render_que, new_weights))
            processes[i].start()
        print(processes)
        for i in range(4*num_episodes):
            task_sequence.put(i)
        k = 0
        state_buffer = list()
        advantage_buffer = list()
        evalue_buffer = list()
        ivalue_buffer = list()
        old_buffer = list()
        imodel_loss_buffer = list()
        usteps = 0
        while True:
            if data.qsize() >= 1:
                self.episode_passed += 1
                k += 1
                (state_memory, ereward_memory, action_memory, old_memory, infox_memory) = data.get()
                steps = len(ereward_memory)
                imodel_loss, irewards =self.get_imodel_result(state_memory, imodel)


                entropy_reward = self.get_entropy_reward(old_memory)
                artificial_reward = irewards #self.get_art_reward(entropy_reward, imodel_loss)


                episode_reward = np.sum(ereward_memory)
                self.scores.append(episode_reward)
                self.rps.append(episode_reward/steps)
                if len(self.mean_score) == 0:
                    self.mean_score.append(episode_reward)
                else:
                    self.mean_score.append(self.mean_score[-1]*0.99 + 0.01 * episode_reward)
                if len(self.rpsm) == 0:
                    self.rpsm.append(self.rps[-1])
                else:
                    self.rpsm.append(self.rpsm[-1]*0.99 + 0.01 * self.rps[-1])

                

                print('----------------------------------------------------------------------------------------------------')
                print('episode: ', self.episode_passed, 'episode_reward: ', episode_reward, 'mean_reward: ', self.mean_score[-1])
                print('global_ent:', self.meanent)

                print('sum_ireward:', np.sum(irewards), 'sum artificial_reward:', np.sum(artificial_reward))
                print('steps:', steps)
                iadvantage_memory, itarget_values = self.compute_advantages(state_memory, artificial_reward, action_memory, iVmodel, self.igamma, self.iw, True, 5)
                print('adv_abs', np.mean(np.abs(iadvantage_memory)))
                print('adv_mean', np.mean(iadvantage_memory))
                
                ereward_memory = self.process_ereward(ereward_memory)
                eadvantage_memory, etarget_values = self.compute_advantages(state_memory, ereward_memory, action_memory, eVmodel, self.egamma, self.ew, True, 50)
                #eadvantage_memory[-25:] = 0


                infox_memory = np.array(infox_memory)
                xpart = np.array([math.floor((x-6736)/6736*1000) for x in infox_memory])  
                for i in range(steps):
                    self.rx[xpart[i]] = self.rx[xpart[i]] * 0.7 + 0.3 *np.sum(irewards[:i])
                    self.arx[xpart[i]] = self.arx[xpart[i]] * 0.7 + 0.3 *np.sum(artificial_reward[:i])
                    self.enx[xpart[i]] = self.enx[xpart[i]] * 0.7 + 0.3 *entropy_reward[i]
                    self.perc[xpart[i]] += 1.0
                state_buffer.extend(state_memory[:-1])
                advantage_buffer.extend(eadvantage_memory+iadvantage_memory)
                evalue_buffer.extend(etarget_values)
                ivalue_buffer.extend(itarget_values)
                old_buffer.extend(old_memory)
                imodel_loss_buffer.extend(imodel_loss)
                usteps = math.ceil(len(state_buffer)/self.batch_size)-1
                print('usteps', usteps)
            if  usteps > self.csteps:
                state_buffer = np.array(state_buffer, dtype=np.uint8)
                advantage_buffer = np.array(advantage_buffer)
                evalue_buffer = np.array(evalue_buffer)
                ivalue_buffer = np.array(ivalue_buffer)
                old_buffer = np.array(old_buffer)
                imodel_loss_buffer = np.array(imodel_loss_buffer)
                    
                agenerator = self.batch_generator(state_buffer, advantage_buffer, evalue_buffer, old_buffer, prioretization=False)
                Amodel.fit_generator(generator=agenerator, 
                                            steps_per_epoch=usteps, 
                                            epochs=self.epochs, 
                                            verbose=1)
                evgenerator = self.batch_generator(state_buffer, advantage_buffer, evalue_buffer, old_buffer, prioretization=False)
                eVmodel.fit_generator(generator=evgenerator, 
                                            steps_per_epoch=usteps, 
                                            epochs=self.epochs, 
                                            verbose=1)

                batch_size = 256
                isteps = math.ceil(len(state_buffer)/batch_size)
                igenerator = self.ireward_batch_generator(state_buffer, imodel_loss_buffer, batch_size)
                imodel.fit_generator(generator=igenerator,
                                            steps_per_epoch=isteps-1, 
                                            epochs=1, 
                                            verbose=1)

                ivgenerator = self.batch_generator(state_buffer, advantage_buffer, ivalue_buffer, old_buffer, prioretization=False)
                    
                iVmodel.fit_generator(generator=ivgenerator, 
                                            steps_per_epoch=usteps, 
                                            epochs=1, 
                                            verbose=1)
                state_buffer = list()
                advantage_buffer = list()
                evalue_buffer = list()
                ivalue_buffer = list()
                old_buffer = list()
                imodel_loss_buffer = list()
                imodel.save_weights(r'E:\sonic_models\imodel.h5')
                Amodel.save_weights(r'E:\sonic_models\policy_model.h5')
                eVmodel.save_weights(r'E:\sonic_models\evalue_model.h5')
                iVmodel.save_weights(r'E:\sonic_models\ivalue_model.h5')
                result_file = open(r'E:\sonic_models\results.json', 'w+')
                score_file = open(r'E:\sonic_models\scores.json', 'w+')
                self.rx = list(self.rx)
                self.arx = list(self.arx)

                if len(self.scores)>1000:
                    temp_scores, temp_mean_score, temp_rps, temp_rpsm = [], [], [], []
                    for i in range(1000):
                        temp_scores.append(self.scores[math.floor(len(self.scores)/1000*i)])
                        temp_mean_score.append(self.mean_score[math.floor(len(self.mean_score)/1000*i)])
                        temp_rps.append(self.rps[math.floor(len(self.rps)/1000*i)])
                        temp_rpsm.append(self.rpsm[math.floor(len(self.rpsm)/1000*i)])
                    self.scores = temp_scores[:]
                    self.mean_score = temp_mean_score[:]
                    self.rps = temp_rps[:]
                    self.rpsm = temp_rpsm[:]
                results['imin'] = self.imin 
                results['imean'] = self.imean
                results['imean2'] = self.imean2
                results['imax'] = self.imax 
                results['istd'] = self.istd
                results['istd2'] = self.istd2
                results['meanent'] = self.meanent
                results['maxent'] = self.maxent
                results['episode_passed'] = self.episode_passed
                sco_results['mean_score'] = self.mean_score
                results['num_episodes'] = num_episodes
                results['tempmax'] = self.tempmax
                results['counts'] = self.counts
                sco_results['scores'] = self.scores
                sco_results['rps'] = self.rps
                sco_results['rpsm'] = self.rpsm
                sco_results['rx'] = self.rx
                sco_results['arx'] = self.arx
                sco_results['enx'] = self.enx
                sco_results['perc'] = self.perc
                json.dump(results, result_file)
                json.dump(sco_results, score_file)
                result_file.close()
                score_file.close()
                for i in range(num_workers):
                    new_weights.put(True)
                if self.render:
                    render_que.put(True)
                for i in range(10*num_episodes):
                    task_sequence.put(i)

        for i in range(num_workers):
            processes[i].join()
        print('all processes performed')

    def batch_generator(self, state_buffer, advantage_buffer, value_buffer, old_buffer, prioretization=False):
        while True:
            if prioretization:
                steps = len(value_buffer)
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

    def ireward_batch_generator(self, state_buffer, old_loss, batch_size):
        while True:
            steps = len(state_buffer)
            #for i in range(math.floor(steps/10)):
            #    state_buffer[10*i] = state_buffer[0]
            randomize = np.arange(len(state_buffer))
            np.random.shuffle(randomize)
            state_buffer = state_buffer[randomize]
            old_loss = old_loss[randomize]
            for ind in range(0, len(state_buffer)-batch_size, batch_size):
                state_batch = np.array(state_buffer[ind:ind+batch_size,:], dtype=np.uint8)
                old_loss_batch = np.array(old_loss[ind:ind+batch_size], dtype=np.float32)
                zero_ireward_batch = np.zeros((batch_size,1))
                yield [state_batch, old_loss_batch], zero_ireward_batch

    def run_episode(self, Amodel, render=False):
        import retro
        env = retro.make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1', scenario='contest')
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
                    env.render(mode='rgb_array', close=True)
                    return state_memory, ereward_memory, action_memory, old_memory, infox

                if random.random() < 0:
                    (temp1, temp2, temp3), cur_policy = self.choose_action(cur_mem, Amodel, render)  
                else:    
                    (action, adesc, adescn), cur_policy = self.choose_action(cur_mem, Amodel, render)
                old_memory.append(cur_policy)
                next_state, reward, done, info = self.wrapped_step(action, env)
                if render:
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
        import PIL
        K.set_floatx('float32')
        K.set_epsilon(1e-9)
        gpu_options = tf.GPUOptions(allow_growth=True)
        K.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))
        policy_net = create_policy_net('E:\sonic_models\policy_model.h5', self.state_len)
        
        while True:
            
            #item = task_sequence.get()
            render = False
            if render_que.qsize()>0:
                render = render_que.get()
            if new_weights.qsize()>0:
                new_weights.get()
                l.acquire()
                try:
                    policy_net.load_weights(r'E:\sonic_models\policy_model.h5')
                    print('new weights are loaded by process', os.getpid())
                except:
                    print('new weights are not loaded by process', os.getpid())
                l.release()
            state_memory, ereward_memory, action_memory, old_memory, infox_memory = self.run_episode(policy_net, render)
            data.put((state_memory, ereward_memory, action_memory, old_memory, infox_memory))

    def process_states(self, state_memory):
        states = np.array(state_memory, dtype=np.uint8)
        states.reshape((len(state_memory), self.state_len, self.height, self.width, 3))
        return states.astype(np.uint8)

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
            print([round(pol,6) for pol in policy])
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
            ereward_memory[-1]=-sume/5
        return ereward_memory

    def compute_advantages(self, state_memory, reward_memory, action_memory, value_model, gamma, weight, trace, multi):
        
        states = self.process_states(state_memory)
        zero_rew = np.zeros((len(state_memory), self.action_space))
        zero_old = np.zeros((len(state_memory), self.action_space))
        if value_model:
            policy_temp, values = value_model.predict([states, zero_rew, zero_old])
            V_current = values[:-1,0]
            V_next = values[1:,0]
        
        G = []
        temp = 0
        for reward in reward_memory[::-1]:
            temp = weight * reward + temp * gamma
            G.append(temp)
        G.reverse()
        G = np.array(G, dtype=np.float32)
        steps = len(states)-1
        actions = np.zeros((steps,self.action_space))
        advantages = np.zeros((steps,self.action_space)).astype(self.adv)
        target_values = np.zeros((steps,1)).astype(self.adv)
        td = np.zeros((steps,1)).astype(self.adv)
        multistep = multi
        if value_model:
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
        else:
            for ind in range(steps):
                td[ind] = G[ind]
                target_values[ind,0] = G[ind]
                advantages[ind, action_memory[ind]] = td[ind]
        if trace:
            if value_model:
                print('vmean:', np.mean(V_current))
            print('meanG:', np.mean(G))
            print('values:', np.mean(target_values))
        return advantages, target_values
                
                
                

            
               



if __name__ == '__main__':
    agent = SonicAgent()
    agent.run_train()



