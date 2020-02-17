
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
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import queue
from multiprocessing import Queue, Process, Lock

def create_reward_net(state_len):
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
    
    def reward_loss():
        def custom_loss(y_true, y_pred):
            return y_pred
        return custom_loss

    def last_image(tensor):
        return tensor[:,-1,:]
    def ireward(tensor):
        return 0.5 * K.mean(K.mean(K.mean(K.square(tensor[0] - tensor[1]), axis=-1), axis=-1), axis=-1)

    state_input = Input(shape=(state_len, height, width, 3))
    float_input = K.cast(state_input, dtype='float32')
    float_input = Lambda(lambda input: input/127.5-1)(float_input)
    float_input = Lambda(last_image)(float_input)
    xs = Conv2D(32, (8,8), padding='same')(float_input)
    xs = LeakyReLU(alpha=0.3)(xs)
    #xs = Activation('relu')(xs)
    xs = MaxPooling2D((4,4))(xs)
    xs = Conv2D(64, (4,4), padding='same')(xs)
    xs = LeakyReLU(alpha=0.3)(xs)
    #xs = Activation('relu')(xs)
    xs = MaxPooling2D((2,2))(xs)
    xs = Conv2D(64, (4,4), strides=(1,1), padding='same')(xs)
    xs = LeakyReLU(alpha=0.3)(xs)
    #xs = Activation('relu')(xs)
    square_size = K.int_shape(xs)
    xs = Flatten()(xs)
    flatten_size = K.int_shape(xs)
    xs = Dense(2048)(xs)
    xs = LeakyReLU(alpha=0.3)(xs)
    xs = Dense(flatten_size[1])(xs)
    xs = LeakyReLU(alpha=0.3)(xs)
    xs = Reshape(square_size[1:])(xs)
    xs = Conv2D(64, (1,1), padding='same')(xs)
    xs = LeakyReLU(alpha=0.3)(xs)
    #xs = Activation('relu')(xs)
    xs = Conv2D(64, (4,4), padding='same')(xs)
    xs = LeakyReLU(alpha=0.3)(xs)
    #xs = Activation('relu')(xs)
    xs = UpSampling2D((2, 2))(xs)
    xs = Conv2DTranspose(32, (4,4), padding='same')(xs)
    xs = LeakyReLU(alpha=0.3)(xs)
    #xs = Activation('relu')(xs)
    xs = UpSampling2D((4, 4))(xs)
    xs = Conv2D(3, (8,8), padding='same')(xs)
    float_output = Activation('linear')(xs)
    intrinsic_reward = Lambda(ireward)([float_input, float_output])
    model = Model(inputs=state_input, outputs=intrinsic_reward)
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss=reward_loss())
    return model

def create_policy_net(model_type, state_len):
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
    height = 88
    width = 120
    crange = 0.2
    action_space = 10

    def sample_loss(reward_input, old_input):
        def custom_loss(y_true, y_pred):
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
    old_policy_input = Input(shape=(action_space,))
    state_input = Input(shape=(state_len, height, width, 3))
    float_input = K.cast(state_input, dtype='float32')
    float_input = Lambda(lambda input: input/127.5-1)(float_input)
    x = TimeDistributed(Conv2D(32, (8,8), padding='same'))(float_input)
    x = TimeDistributed(Activation('relu'))(x)
    x = TimeDistributed(MaxPooling2D((4,4)))(x)
    x = TimeDistributed(Conv2D(64, (4,4), padding='same'))(x)
    x = TimeDistributed(Activation('relu'))(x)
    x = TimeDistributed(MaxPooling2D((2,2)))(x)
    x = TimeDistributed(Conv2D(64, (4,4), strides=(1,1), padding='same'))(x)
    x = TimeDistributed(Activation('relu'))(x)
    x = TimeDistributed(Flatten())(x)
    x = Lambda(last_image)(x)
    x = Dense(512)(x)
    x = Activation('tanh')(x)

    adam1 = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0, amsgrad=False)
    if model_type=='policy':
        main_output = Dense(action_space, activation='softmax', name='main_output')(x)
        model = Model(inputs=[state_input,reward_input,old_policy_input],outputs=main_output)
        model.compile(optimizer=adam1, loss=sample_loss(reward_input,old_policy_input))
    elif model_type=='critic':
        critic_output = Dense(1, activation='linear', name='critic_output')(x)
        model = Model(inputs=state_input,outputs=critic_output)
        model.compile(optimizer=adam1, loss=critic_loss)
    else:
        raise Exception('model type is not named')
    
    return model
    

class SonicAgent():
    def __init__(self, episodes_step=4, max_env_steps=None, state_len=1, gamma=0.99, batch_size=256, workers=6, render=True):
        self.timedelay = 10
        self.batch_size = batch_size
        self.csteps = 8 * 36000 / self.timedelay / self.batch_size
        self.workers = workers
        self.iw = 1
        self.ew = 0
        self.epochs = 1
        self.action_space = 10
        self.actions = self.get_actions()
        self.state_len = 1
        self.height = 88
        self.width = 120
        self.gamma = gamma
        self.egamma = gamma
        self.igamma = gamma
        self.episodes = episodes_step
        self.ignored_steps = 15
        self.render = render
        self.default = np.float32
        self.adv = np.float32
        self.stats = dict()
        # self.special_game_ids = [2**i for i in range(20)]
        # temp = []
        # for num in self.special_game_ids[:]:
        #     for i in range(0,4):
        #         temp.append(num+i)
        # self.special_game_ids = temp[:]
        


    def get_entropy(self, policy_sequence):
        policy_array = np.array(policy_sequence)
        entropy = -np.sum(policy_array * np.log(policy_array), axis=-1) / (-np.log(1/self.action_space))
        return entropy
    
    def get_imodel_result(self, state_memory, imodel_new, imodel_old):
        steps = len(state_memory)
        states = self.process_states(state_memory)
        loss_old = imodel_old.predict(states)
        loss_new = imodel_new.predict(states)
        ratio = loss_new / loss_old
        ratio /= self.stats['istd'] * 10
        irewards = np.zeros((steps-1,))
        temp = ratio[0]
        for i in range(1,steps):
            if ratio[i]>temp:
                irewards[i-1] = ratio[i] - temp
                temp = ratio[i]
        irewards[-self.ignored_steps:] = 0

        imean = np.mean(ratio)
        istd = np.std(ratio)
        imin = np.min(ratio)
        imax = np.max(ratio)
        m = self.stats['window'] * 36000/ self.timedelay
        if self.stats['episodes_passed'] <= 1:
            self.stats['imean'] = imean
            self.stats['istd'] = istd
            self.stats['imax'] = imax
            self.stats['imin'] = imin
        self.stats['imean'] = ((m - steps) * self.stats['imean'] + steps * imean)/m
        self.stats['istd']  = ((m - steps) * self.stats['istd'] + steps * istd)/m
        self.stats['imax'] = ((m - steps) * self.stats['imax'] + steps * imax)/(m)
        self.stats['imin'] = ((m - steps) * self.stats['imin'] + steps * imin)/(m)
        return irewards, ratio

    def get_art_reward(self,entropy_reward, imodel_loss):
        multi = np.array(list(map(lambda x:1e60**x, imodel_loss-imodel_loss[0])))
        multi /= 10*self.stats['istd']
        artificial_reward = multi[1:] - multi[:-1]
        return artificial_reward

    def get_stats(self, path=r'D:\sonic_models\results.json'):
        try:
            with open(path, 'r') as file:
                self.stats = json.load(file)
        except:
            print('loading default stats...')
            self.stats['episodes_passed'] = 0
            self.stats['imean'] = 1
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


    def save_stats(self, path=r'D:\sonic_models\results.json'):
        try:
            with open(path, 'w') as file:
                json.dump(self.stats, file, indent=4)
            print('Stats saved')
        except:
            print('Error occured for saving stats')

    def run_train(self):
        lock = Lock()
        lock.acquire()
        self.get_stats()
        
        
        num_workers = self.workers
        data = Queue()
        task_sequence = Queue()
        new_weights = Queue()
        render_que = Queue()
        render_que.put(True)

        Amodel = create_policy_net('policy', self.state_len)
        if os.path.isfile(r'D:\sonic_models\policy_model.h5'):
            Amodel.load_weights(r'D:\sonic_models\policy_model.h5')

        eVmodel = create_policy_net('critic', self.state_len)
        if os.path.isfile(r'D:\sonic_models\evalue_model.h5'):
            eVmodel.load_weights(r'D:\sonic_models\evalue_model.h5')

        iVmodel = create_policy_net('critic', self.state_len)
        if os.path.isfile(r'D:\sonic_models\ivalue_model.h5'):
            iVmodel.load_weights(r'D:\sonic_models\ivalue_model.h5')

        imodel_new = create_reward_net(self.state_len)
        if os.path.isfile(r'D:\sonic_models\imodel_new.h5'):
            imodel_new.load_weights(r'D:\sonic_models\imodel_new.h5')
        
        imodel_old = create_reward_net(self.state_len)
        if os.path.isfile(r'D:\sonic_models\imodel_old.h5'):
            imodel_old.load_weights(r'D:\sonic_models\imodel_old.h5')
        lock.release()


        processes = []
        for i in range(num_workers):
            processes.append(Process(target=self.worker, args=(lock, data, task_sequence, render_que)))
            processes[-1].start()
        print(processes)
        for game_id in range(self.stats['episodes_passed'] + 1, 50000):
            task_sequence.put(game_id)
        state_buffer = list()
        advantage_buffer = list()
        evalue_buffer = list()
        ivalue_buffer = list()
        old_buffer = list()
        usteps = 0
        while True:
            if data.qsize() >= 1:
                self.stats['episodes_passed'] += 1
                (state_memory, ereward_memory, action_memory, old_memory, coords) = data.get()
                steps = len(ereward_memory)
                irewards, ratio = self.get_imodel_result(state_memory, imodel_new, imodel_old)
                entropy = self.get_entropy(old_memory)
                self.stats['entropy'].append(np.mean(entropy))
                self.stats['mean100_entropy'].append(np.mean(self.stats['entropy'][-100:]))
                self.stats['external_rewards'].append(np.sum(ereward_memory))
                self.stats['external_rewards_per_step'].append(np.sum(ereward_memory)/steps)
                self.stats['mean100_external_rewards'].append(np.mean(self.stats['external_rewards'][-100:]))
                
                print('---------------------------------------------------------')
                print('episode: ', self.stats['episodes_passed'],
                    'episode_reward: ', self.stats['external_rewards'][-1],
                    'mean_reward: ', self.stats['mean100_external_rewards'][-1])
                print('mean100_entropy:', self.stats['mean100_entropy'][-1])
                print('sum_ireward:', np.sum(irewards))
                print('steps:', steps)

                iadvantage_memory, itarget_values = self.compute_advantages(state_memory, irewards, action_memory, iVmodel, self.igamma, self.iw, False, 3600)
                ereward_memory = np.array(ereward_memory)
                eadvantage_memory, etarget_values = self.compute_advantages(state_memory, ereward_memory, action_memory, eVmodel, self.egamma, self.ew, True, 3600)
                
                self.stats['x'].extend(coords['x'][:-self.ignored_steps])
                self.stats['y'].extend(coords['y'][:-self.ignored_steps])
                self.stats['some_map'].extend(list(ratio[1:-self.ignored_steps-1]))
                self.stats['ireward_map'].extend(list(irewards[1:-self.ignored_steps-1]))
                self.stats['entropy_map'].extend(list(entropy[1:-self.ignored_steps-1]))


                state_buffer.extend(state_memory[:-1])
                advantage_buffer.extend(iadvantage_memory)
                evalue_buffer.extend(etarget_values)
                ivalue_buffer.extend(itarget_values)
                old_buffer.extend(old_memory)
                usteps = math.ceil(len(state_buffer)/self.batch_size)-1
                print('usteps', usteps)
                print('----------------------------------------------------------------------------------------------------')
            if  usteps > self.csteps:
                state_buffer = np.array(state_buffer, dtype=np.uint8)
                advantage_buffer = np.array(advantage_buffer)
                evalue_buffer = np.array(evalue_buffer)
                ivalue_buffer = np.array(ivalue_buffer)
                old_buffer = np.array(old_buffer)

                if self.stats['episodes_passed'] >50:
                    print('training policy model...')
                    agenerator = self.batch_generator(state_buffer, advantage_buffer, evalue_buffer, old_buffer, target='policy', prioretization=True)
                    Amodel.fit_generator(generator=agenerator, 
                                                steps_per_epoch=usteps, 
                                                epochs=self.epochs, 
                                                verbose=1)
                # print('training ext value model...')
                # evgenerator = self.batch_generator(state_buffer, advantage_buffer, evalue_buffer, old_buffer, target='value', prioretization=False)
                # eVmodel.fit_generator(generator=evgenerator, 
                #                             steps_per_epoch=usteps, 
                #                             epochs=self.epochs, 
                #                             verbose=1)
                print('training ireward new model...')                            
                igenerator = self.ireward_batch_generator(state_buffer, 64)
                imodel_new.fit_generator(generator=igenerator,
                                            steps_per_epoch=200, 
                                            epochs=1, 
                                            verbose=1)
                print('training ireward old model...')   
                imodel_old.fit_generator(generator=igenerator,
                                            steps_per_epoch=5, 
                                            epochs=1, 
                                            verbose=1)
                print('training int value model...')
                ivgenerator = self.batch_generator(state_buffer, advantage_buffer, ivalue_buffer, old_buffer, target='value', prioretization=False)
                iVmodel.fit_generator(generator=ivgenerator, 
                                           steps_per_epoch=usteps, 
                                           epochs=1, 
                                           verbose=1)
                state_buffer = list()
                advantage_buffer = list()
                evalue_buffer = list()
                ivalue_buffer = list()
                old_buffer = list()

                lock.acquire()
                imodel_new.save_weights(r'D:\sonic_models\imodel_new.h5')
                imodel_old.save_weights(r'D:\sonic_models\imodel_old.h5')
                Amodel.save_weights(r'D:\sonic_models\policy_model.h5')
                eVmodel.save_weights(r'D:\sonic_models\evalue_model.h5')
                iVmodel.save_weights(r'D:\sonic_models\ivalue_model.h5')

                
                for key in self.stats.keys():
                    if isinstance(self.stats[key], list):
                        self.stats[key] = list(map(lambda x: float(x), self.stats[key]))
                    elif isinstance(self.stats[key], np.generic):
                        self.stats[key] = float(self.stats[key])
                self.save_stats()

                self.stats['x'] = list()
                self.stats['y'] = list()
                self.stats['some_map'] = list()
                self.stats['ireward_map'] = list()
                self.stats['entropy_map'] = list()

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



    def batch_generator(self, state_buffer, advantage_buffer, value_buffer, old_buffer, target, prioretization=False):
        steps = len(value_buffer)
        p_small = np.array([(1/(i+1))**0.5 for i in range(steps)])
        p_big = p_small/np.sum(p_small)
        indes = np.argsort(np.max(np.abs(advantage_buffer), axis=-1), axis=0)[::-1]
        allen =math.ceil(steps/self.batch_size) * self.batch_size * self.epochs
        w = 1 / allen / p_big
        w /= np.max(w)
        while True:
            if prioretization:
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
                    dummy_action_batch = np.zeros((self.batch_size,self.action_space))
                    if target=='policy':
                        yield [state_batch, advantage_batch,  old_batch], dummy_action_batch
                    elif target=='value':
                        yield state_batch, value_batch

    def ireward_batch_generator(self, state_buffer, batch_size):
        while True:
            steps = len(state_buffer)
            randomize = np.arange(len(state_buffer))
            np.random.shuffle(randomize)
            state_buffer = state_buffer[randomize]
            for ind in range(0, len(state_buffer)-batch_size, batch_size):
                state_batch = np.array(state_buffer[ind:ind+batch_size,:], dtype=np.uint8)
                dummy_target = np.zeros((batch_size,1))
                yield state_batch, dummy_target

    def run_episode(self, Amodel, render=False, record=False, game_id=0, path='.'):
        import retro
        if record:
            env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1', scenario='contest', record=path)    
        else:
            env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1', scenario='contest')
        env.movie_id = game_id
        cur_mem = deque(maxlen = self.state_len)
        done = False
        delay_limit = self.timedelay
        episode_reward = 0
        control_count = 0
        frame_reward = 0
        state_memory = []
        ereward_memory = []
        action_memory = []
        old_memory = []
        coords = {'x': [], 'y': []}
        for j in range(self.state_len):
            cur_mem.append(np.zeros((self.height, self.width, 3),dtype=self.default))
        state = env.reset()
        state = self.resize_state(state)
        cur_mem.append(np.copy(state))
        state_memory.append(np.copy(cur_mem))
        action_id, cur_policy = self.choose_action(cur_mem, Amodel, render)
        old_memory.append(cur_policy)
        while True:
            if (control_count == delay_limit) or done:
                state_memory.append(np.copy(cur_mem))
                action_memory.append(action_id)
                ereward_memory.append(frame_reward)
                if done:
                    env.render(mode='rgb_array', close=True)
                    return state_memory, ereward_memory, action_memory, old_memory, coords

                # if random.random() < 0:
                #     temp, cur_policy = self.choose_action(cur_mem, Amodel, render)  
                action_id, cur_policy = self.choose_action(cur_mem, Amodel, render)
                old_memory.append(cur_policy)
                next_state, reward, done, info = env.step(self.actions[action_id])
                coords['x'].append(info['x'])
                coords['y'].append(info['y'])
                next_state = self.resize_state(next_state)
                cur_mem.append(np.copy(next_state))
                if render:
                    env.render()
                frame_reward = 0
                control_count = 0
            else:
                next_state, reward, done, info = env.step(self.actions[action_id])
            control_count += 1
            frame_reward += reward

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
        
        policy_net = create_policy_net('policy', self.state_len)
        if os.path.isfile(r'D:\sonic_models\policy_model.h5'):
            policy_net.load_weights(r'D:\sonic_models\policy_model.h5')
            policy_loading_time = os.path.getmtime('D:\sonic_models\policy_model.h5')
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
                if policy_loading_time<os.path.getmtime('D:\sonic_models\policy_model.h5'):
                    policy_loading_time = os.path.getmtime('D:\sonic_models\policy_model.h5')
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
        target = [0] * self.action_space
        buttons = ('B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z')
        actions = [[], ['LEFT'], ['RIGHT'], ['LEFT','DOWN'], ['RIGHT','DOWN'], ['DOWN'], ['DOWN', 'B'], ['B'], ['LEFT','B'], ['RIGHT','B']]
        for i in range(self.action_space):
            action_list = [0] * len(buttons)
            for button in actions[i]:
                action_list[buttons.index(button)]=1
            target[i]=np.array(action_list)
        return target    
        
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

    def compute_advantages(self, state_memory, reward_memory, action_memory, value_model, gamma, weight, trace, multi):
        
        states = self.process_states(state_memory)
        if value_model:
            values = value_model.predict(states)
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
    # result_file = open(r'D:\sonic_models\results.json', 'r')
    # results = json.load(result_file)
    # episode_passed = results['episode_passed']
    # Amodel = create_policy_net('policy', 1)
    # if os.path.isfile(r'D:\sonic_models\policy_model.h5'):
    #     Amodel.load_weights(r'D:\sonic_models\policy_model.h5')
    # agent.run_episode(Amodel, render=True, record=True, game_id=episode_passed, path=r'D:\sonic_models\replays')


