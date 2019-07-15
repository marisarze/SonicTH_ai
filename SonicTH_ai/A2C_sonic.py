
import os
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
from multiprocessing import Queue, Process


#class ThreadWithReturnValue(Thread):
#    def __init__(self, group=None, target=None, name=None,
#                 args=(), kwargs={}, Verbose=None):
#        Thread.__init__(self, group, target, name, args, kwargs)
#        self._return = None
#    def run(self):
#        print(type(self._target))
#        if self._target is not None:
#            self._return = self._target(*self._args,
#                                                **self._kwargs)
#    def join(self, *args):
#        Thread.join(self, *args)
#        return self._return

def create_policy_net(path, state_len=1):
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
    from keras import losses
    from keras import regularizers
    from keras.models import Model
    from keras import backend as K
    from keras.losses import mean_squared_error
    from keras.losses import categorical_crossentropy
    
    state_len = state_len
    height = 90
    width = 128
    usb = False
    crange = 0.2
    bm = 0.99
    action_space = 10
    adam1 = Adam(lr=2e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-5, decay=0.0, amsgrad=False, clipvalue=5.0)
    #adam2 = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0, clipnorm=10., amsgrad=False)
    #adam1 = SGD(lr=2e-4, momentum=0.0, decay=0.0, nesterov=False, clipvalue=50.0)
    #adam2 = SGD(lr=1e-2, momentum=0.0, decay=0.0, nesterov=False)
    def sample_loss(reward_input, old_input):
        def custom_loss(y_true, y_pred):
            ratio = y_pred/(old_input+1e-4)
            pg_loss1 = -reward_input * ratio
            pg_loss2 = -reward_input * K.clip(ratio, 1.0 - crange, 1.0 + crange)
            pg_loss = K.maximum(pg_loss1,pg_loss2)
            return K.mean(pg_loss,axis=-1)
            #return -K.mean(K.sum(reward_input * K.log(y_pred+1e-3), axis=1), axis=0)# + K.mean(K.sum(0.05 * K.square(y_pred/(old_input)-1),axis=1),axis=0)
        return custom_loss

    def critic_loss(y_true, y_pred):
        return K.mean(0.5 * K.square(y_true - y_pred), axis=-1)
    def last_image(tensor):
        return tensor[:,-1,:]

    main_input = Input(shape=(state_len, height, width, 3))
    xm = TimeDistributed(Conv2D(40, (8,8), padding='same', use_bias=usb))(main_input)
    xm = TimeDistributed(BatchNormalization(momentum=bm))(xm)
    xm = TimeDistributed(Activation('tanh'))(xm)
    #xm = TimeDistributed(PReLU())(xm)
    xm = TimeDistributed(MaxPooling2D((4,4)))(xm)
    xm = TimeDistributed(Conv2D(80, (6,6), padding='same', use_bias=usb))(xm)
    xm = TimeDistributed(BatchNormalization(momentum=bm))(xm)
    xm = TimeDistributed(Activation('tanh'))(xm)
    #xm = TimeDistributed(PReLU())(xm)
    xm = TimeDistributed(MaxPooling2D((3,3)))(xm)
    xm = TimeDistributed(Conv2D(80, (6,6), padding='same', use_bias=usb))(xm)
    xm = TimeDistributed(BatchNormalization(momentum=bm))(xm)
    xm = TimeDistributed(Activation('tanh'))(xm)
    #xm = TimeDistributed(PReLU())(xm)
    xm = TimeDistributed(MaxPooling2D((3,3)))(xm)
    xm = TimeDistributed(Flatten())(xm)
    #xmr = CuDNNLSTM(50, return_sequences=False)(xm)
    #xm = TimeDistributed(BatchNormalization())(xm)
    #xmr = CuDNNLSTM(10, return_sequences=False)(xm)
        
    xm1= Lambda(last_image)(xm)

    #features = Concatenate()([xm1,xmr])
    features = BatchNormalization(momentum=bm)(xm1)
    feature_dimension = int(features.shape[1])
    print('feature shape is: ', feature_dimension)
        
    xm = Dense(feature_dimension, use_bias=usb)(features)
    xm = BatchNormalization(momentum=bm)(xm)
    xm = Activation('tanh')(xm)
    #xm = PReLU()(xm)
    #xm = BatchNormalization()(xm)
    #feature_dimension = math.ceil(feature_dimension/2)
    xm = Dense(feature_dimension, use_bias=usb)(xm)
    xm = BatchNormalization(momentum=bm)(xm)
    xm = Activation('tanh')(xm)
    #xm = PReLU()(xm)
    reward_input = Input(shape=(action_space,))
    old_input = Input(shape=(action_space,))
    #xm = BatchNormalization()(xm)
    xm = Dense(action_space,  use_bias=usb)(xm)
    xm = BatchNormalization(momentum=bm)(xm)
    main_output = Activation('softmax', name='main_output')(xm)

    #xc = BatchNormalization()(xm1)
    xc = Dense(feature_dimension, use_bias=usb)(features)
    xc = BatchNormalization(momentum=bm)(xc)
    xc = Activation('tanh')(xc)
    #xc = PReLU()(xc)
        
    xc = Dense(1, use_bias=usb)(xc)
    xc = BatchNormalization(momentum=bm)(xc)
    critic_output = Activation('sigmoid', name='critic_output')(xc)
    #critic_output = PReLU(name='critic_output')(xc)
    model = Model(inputs=[main_input,reward_input,old_input],outputs=[main_output, critic_output])

    if path=='policy_model.h5':
        weights={'main_output': 1, 'critic_output': 0}
    elif path=='value_model.h5':
        weights={'main_output': 0, 'critic_output': 1}

    exists = os.path.isfile(path)
    if exists:
        model.load_weights(path)
        print('weights loaded from '+path)
    model.compile(optimizer=adam1, 
                loss={'main_output': sample_loss(reward_input,old_input), 'critic_output': critic_loss}, 
                loss_weights=weights)
    #model.summary()
    return model
    

class SonicAgent():
    def __init__(self, n_episodes=50000, max_env_steps=None, state_len=1, gamma=0.99, batch_size=32):
        
        self.epochs = 1
        self.action_space = 10
        self.state_len = state_len
        self.epsilon_max = 1
        self.batch_size = batch_size
        self.height = 90
        self.width = 128
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.beta = 0.1
        self.timedelay = 12
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



    def choose_action(self, state_col, Amodel, Vmodel):
        states = np.reshape(np.array(state_col)/127-1, (1, len(state_col), self.height, self.width, 3)).astype(self.default)
        zero_rew = np.zeros((1, self.action_space))
        zero_old = np.zeros((1, self.action_space))
        policy, value_temp = Amodel.predict([states, zero_rew, zero_old])
        policy_temp, value = Vmodel.predict([states, zero_rew, zero_old])
        policy = policy[-1]
        #print([round(policy[i],6) for i in range(len(policy))], 'steps: ', self.steps, 'value: ', repr(V[0,0]), 'buflen: ', len(self.value_buffer))
        if random.random() < 0:#0.05 + 0.95*math.exp(-self.training_steps / self.tau):
            order = random.randint(0,9)
            #print('random: ', order, 'steps: ', self.steps)
            order = self.convert_order(order)
        else:
            if np.max(policy) >= 0.99:
                order = self.convert_order(np.argmax(policy))
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
        V_current=np.array(value_memory[:-1], dtype=np.float64)
        V_next=np.array(value_memory[1:], dtype=np.float64)
        steps = len(value_memory)
        actions = np.zeros((steps,self.action_space))
        Advantage = np.zeros((steps,self.action_space)).astype(self.adv)
        V_target = np.zeros((steps,1)).astype(self.adv)
        td = np.zeros((steps,1)).astype(self.adv)
        multistep = 50
        for ind in range(steps):
            if ind+multistep < steps:
                multirew = 0
                for rew in reversed(reward_memory[ind:ind+multistep]):
                    multirew = rew + self.gamma * multirew
                td[ind] = multirew + self.gamma**(multistep) * V_next[ind+multistep-1] - V_current[ind]
                V_target[ind,0] = multirew + self.gamma**(multistep) * V_next[ind+multistep-1]
            else:
                td[ind] = G[ind] - value_memory[ind]
                V_target[ind,0] = G[ind]
                Advantage[ind, action_memory[ind]] = td[ind]
        print('mean: ', np.mean(np.abs(td)/(np.array(value_memory)+1e-8)), 'max: ', np.max(np.abs(td)/(np.array(value_memory)+1e-8)), 'min: ', np.min(np.abs(td)/(np.array(value_memory)+1e-8)))
        return Advantage, V_target
        
        #td[ind] = self.R_memory[ind]
        #Advantage[ind, a[ind]] = td[ind]
        #Advantage -= np.mean(Advantage)
        #Advantage /= np.std(Advantage)
        
    def run_episode(self, episode, Amodel, Vmodel, render=False):
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
        reward_memory = []
        action_memory = []
        old_memory = []
        value_memory = []
        for j in range(self.state_len):
            cur_mem.append(np.zeros((self.height, self.width, 3),dtype=self.default))
        state = self.wrapped_reset(env)
        cur_mem.append(np.copy(state))
        state_memory.append(np.copy(cur_mem))
        (action, adesc, adescn), cur_policy, cur_value = self.choose_action(cur_mem, Amodel, Vmodel)
        old_memory.append(cur_policy)
        value_memory.append(cur_value)
        while True:
            if (control_count == self.timedelay) or done:
                state_memory.append(np.copy(cur_mem))
                action_memory.append(adescn)
                reward_memory.append(frame_reward)
                if done:
                    return state_memory, reward_memory, action_memory, old_memory, value_memory

                if random.random() < 0.0:
                    (temp1, temp2, temp3), cur_policy, cur_value = self.choose_action(cur_mem, Amodel, Vmodel)  
                else:    
                    (action, adesc, adescn), cur_policy, cur_value = self.choose_action(cur_mem, Amodel, Vmodel)
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

    def batch_generator(self, prioretization=False, alpha=1):
        while True:
            if prioretization:
                steps = len(self.value_buffer)
                p_small = np.array([(1/(i+1))**alpha for i in range(steps)])
                p_big = p_small/np.sum(p_small)
                indes = np.argsort(np.max(np.abs(self.advantage_buffer), axis=-1), axis=0)[::-1]
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
                    state_batch0.append(self.state_buffer[targ])
                    advantage_batch0.append(w[ind] * self.advantage_buffer[targ,:])
                    old_batch0.append(self.old_buffer[targ,:])
                    value_batch0.append(self.value_buffer[targ,:])
                    i += 1
                    if len(state_batch0) == self.batch_size:
                        state_batch = np.array(state_batch0, dtype=np.float32)/127 - 1
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
                randomize = np.arange(len(self.state_buffer))
                np.random.shuffle(randomize)
                #print('st_buf:',len(self.state_buffer))
                #print('advantage_buf:',len(self.advantage_buffer))
                #print('value_buf:',len(self.value_buffer))
                #print('old_buf:',len(self.old_buffer))
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
            




    def worker(self, path, data, task_sequence, render_que, new_weights, use_gpu=True):
        import os
        if not use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        import sys
        from keras import backend as K
        K.set_epsilon(1e-9)
        gpu_options = tf.GPUOptions(allow_growth=True)
        K.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
        policy_net = create_policy_net('policy_model.h5')
        value_net = create_policy_net('value_model.h5')
        while True:
            item = task_sequence.get()
            render = False
            if render_que.qsize()>0:
                render = render_que.get()
            if new_weights.qsize()>0:
                if new_weights.get():
                    try:
                        policy_net.load_weights('policy_model.h5')
                        value_net.load_weights('value_model.h5')
                        print('new weights are loaded by process', os.getpid())
                    except:
                        print('new weights are not loaded by process', os.getpid())
            #if item == 0:
            #    render = True
            state_memory, reward_memory, action_memory, old_memory, value_memory = self.run_episode(item, policy_net, value_net, render)
            advantage_memory, target_values = self.compute_advantages(reward_memory, action_memory, value_memory)
            data.put((state_memory[:-1], advantage_memory, target_values, old_memory, sum(reward_memory)))

        #print('process ', os.getpid(), 'done')
        #data.close()
        #data.cancel_join_thread()
        #task_sequence.close()
        #task_sequence.cancel_join_thread()

    def run_train(self):
        num_episodes = 8
        num_workers = 8
        self.scores= []
        self.score100 = deque(maxlen = 100)
        self.mean_score = []
        path = None
        data = Queue()
        task_sequence = Queue()
        new_weights = Queue()
        render_que = Queue()
        render_que.put(True)
        for i in range(num_episodes):
            task_sequence.put(i)
        processes = [0] * num_workers
        process_statuses = [None] * num_workers
        pids = [None] * num_workers
        for i in range(num_workers):
            processes[i] = Process(target=self.worker, args=(path, data, task_sequence, render_que, new_weights))
            processes[i].start()
            pids[i] = processes[i].pid
        print(processes)

        m = 0
        k = 0
        state_buffer = list()
        advantage_buffer = list()
        value_buffer = list()
        old_buffer = list()
        from keras import backend as K
        K.set_epsilon(1e-9)
        gpu_options = tf.GPUOptions(allow_growth=True)
        K.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
        Amodel = create_policy_net('policy_model.h5')
        Vmodel = create_policy_net('value_model.h5')
        while True:
            if data.qsize() >= 1:
                m += 1
                k += 1
                (states, advantages, values, olds, episode_reward) = data.get()
                self.scores.append(episode_reward)
                self.score100.append(episode_reward)
                self.mean_score.append(np.mean(self.score100))
                print('episode: ', m, 'episode_reward: ', episode_reward, 'mean_reward: ', self.mean_score[-1])
                print(np.std(np.array(olds), axis=0)/np.mean(np.array(olds), axis=0))
                print('----------------------------------------------------------------------------------------------------')
                state_buffer.extend(states)
                advantage_buffer.extend(advantages)
                value_buffer.extend(values)
                old_buffer.extend(olds)
            if  k == num_episodes:

                #for i in range(num_workers):
                #    processes[i].terminate()

                self.state_buffer = np.array(state_buffer, dtype=np.uint8)
                state_buffer = list()
                self.advantage_buffer = np.array(advantage_buffer)
                #self.advantage_buffer -= np.mean(self.advantage_buffer)
                #self.advantage_buffer -= np.std(self.advantage_buffer)
                advantage_buffer = list()
                self.value_buffer = np.array(value_buffer)
                value_buffer = list()
                self.old_buffer = np.array(old_buffer)
                old_buffer = list()
                steps = math.ceil(len(self.value_buffer)/self.batch_size)-1
                #if nn >=1:
                Amodel.fit_generator(generator=self.batch_generator(prioretization=False, alpha=0.2), 
                                            steps_per_epoch=steps, 
                                            epochs=self.epochs, 
                                            verbose=1)
                Amodel.save_weights('policy_model.h5')
                Vmodel.fit_generator(generator=self.batch_generator(prioretization=False), 
                                            steps_per_epoch=steps, 
                                            epochs=1, 
                                            verbose=1)
                Vmodel.save_weights('value_model.h5')
                k = 0
                render_que.put(True)
                for i in range(num_workers):
                    new_weights.put(True)
                    #processes[i] = Process(target=self.worker, args=(path, data, task_sequence, render_que, new_weights))
                    #processes[i].start()
                for i in range(num_episodes):
                    task_sequence.put(i)




        for i in range(num_workers):
            processes[i].join()
        #while True:
        #    for i in range(num_workers):
        #        print(processes[i].pid, ': ', processes[i].exitcode)
            


        print('all processes performed')



        #import tensorflow as tf
        #from keras import backend as K
        #with tf.Session() as sess:
        #    K.set_session(sess)
        #    sess.run(tf.global_variables_initializer())
        #    graph = tf.get_default_graph()
        #    with graph.as_default():
                
                
                

            
               



if __name__ == '__main__':
    agent = SonicAgent()
    agent.run_train()



         

    
        #print('--------------------------fit main model--------------------------------------')
        #prior_state = []
        #prior_A = []
        #prior_Vtar = []
        #prior_old = []

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