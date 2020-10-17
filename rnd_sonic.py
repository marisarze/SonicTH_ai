
import os
import sys
import json
import random
import math
import numpy as np
import random
import time
import cv2
import gc
from sys import getsizeof
from collections import deque
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import queue

from multiprocessing import Queue, Process, Lock
from rnd_models import *



class SonicAgent():
    def __init__(self, episodes_step=10, max_env_steps=None, state_len=1, gamma=1, batch_size=64, workers=5, render=False):
        self.game_name = 'SonicTheHedgehog-Genesis'
        self.map_name = 'LabyrinthZone.Act2'
        self.scenario = 'contest'
        self.timedelay = 12
        self.batch_size = batch_size
        self.csteps = 36000 / self.timedelay
        self.num_workers = workers
        self.iw = 1.0
        self.ew = 0.0
        self.epochs = 5
        self.count = 5
        self.actions = self.get_actions()
        self.action_space = len(self.actions)
        self.state_len = state_len
        self.width = 120
        self.height = 84
        self.state_shape = (self.state_len, self.height, self.width,3)
        self.lam = 0.999
        self.crange = 0.002
        self.epsilon = 0.1
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
        self.epoch_size = 10
        self.nonusable_steps = 0#math.ceil(math.log(1e-4, self.igamma))
        self.choosed_length = 0 * self.csteps + self.nonusable_steps
        self.memory_size = 10*self.csteps + self.nonusable_steps
        self.horizon = 15 * self.csteps

        self.stats['policy_lr'] = 1e-6
        self.stats['evmodel_lr'] = 1e-6 * math.sqrt(self.batch_size)
        self.stats['ivmodel_lr'] = 1e-6 * math.sqrt(self.batch_size)
        self.stats['imodel_lr'] = 1e-8 
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
    
    def get_irewards(self, states):
        states = self.process_states(states)
        losses = self.get_iloss(states)
        
        if self.stats['initialized'] > 0:
            #losses -= self.stats['imean']
            losses /= 10 * self.stats['istd'] #* self.csteps
        #irewards = losses
        irewards = np.zeros(losses.shape)
        irewards[:-1] = losses[1:] - losses[:-1]
        irewards[-1] = 0 #losses[0] - losses[-1] 
        #irewards *= 1-self.igamma
        #mratio = np.clip(ratio, 0, 100)
        #irewards = self.iw * (1-self.igamma) * ratio
        #irewards = np.clip(irewards, 0, 10)
        return irewards

    def run_train(self):
        lock = Lock()
        self.get_stats()
        data = Queue()
        game_ids = Queue()
        render_que = Queue()
        render_que.put(True)

        for i in range(self.stats['episodes_passed'], 1000000):
            game_ids.put(i)
        
        processes = []
        for i in range(self.num_workers):
            processes.append(Process(target=self.worker, args=(lock, data, render_que, game_ids)))
        
        for p in processes:
            p.start()
        
        print(processes)
        
        
        self.create_models()
        self.reusable_memory = []
        self.base_keys = {'states': np.uint8,
                    'erewards': np.float32,
                    'actions': np.uint8,
                    'coords': np.int32}
        self.all_keys = {'states': np.uint8,
                        'erewards': np.float32,
                        'actions': np.uint8,
                        'coords': np.int32,
                        'irewards': np.float32,
                        'eadvantages': np.float32,
                        'iadvantages': np.float32,
                        'evariance': np.float32,
                        'ivariance': np.float32,
                        'policy': np.float32,
                        'etargets': np.float32,
                        'itargets': np.float32,
                        'advantages': np.float32}
        
        
        while True:
            print('-------------------------------------------------------------------')
            self.clear_maps()
            self.reinitialize_buffer()
            self.current_memory = []
            
            
            
            result = data.get()
            temp_dict = dict()
            for key in self.base_keys.keys():
                result[key] = np.array(result[key], dtype=self.base_keys[key])
                #print(key, result[key].shape)
            self.reusable_memory.append(result)
            result['irewards'] = self.get_irewards(result['states'])
            self.update_stats(result)



            self.reduce_memory()
            fullness = self.get_memory_length() / self.memory_size
            print('memory_fullness', fullness, 'memory_len', len(self.reusable_memory), 'size', getsizeof(self.reusable_memory)/1024**2)
            

            if self.get_memory_length() > self.choosed_length:#len(self.reusable_memory)>0
                self.mean_adv = []
                for result in self.reusable_memory:
                    result['irewards'] = self.get_irewards(result['states'])
                    #result['eadvantages'] = np.zeros((steps, self.action_space))
                    #result['etargets'] = np.zeros((steps, self.action_space))
                    result['eadvantages'], result['etargets'], result['evariance'] = self.compute_advantages(result['states'],
                                                                                                            result['erewards'],
                                                                                                            result['actions'],
                                                                                                            self.evmodel,
                                                                                                            self.egamma,
                                                                                                            self.lam,
                                                                                                            episodic=True,
                                                                                                            trace=False)
                    result['iadvantages'], result['itargets'], result['ivariance'] = self.compute_advantages(result['states'],
                                                                                                            result['irewards'],
                                                                                                            result['actions'],
                                                                                                            self.ivmodel,
                                                                                                            self.igamma,
                                                                                                            self.lam,
                                                                                                            episodic=True,
                                                                                                            trace=False)

                    #self.mean_adv.append(np.mean(np.abs(np.sum(result['eadvantages'], axis=-1))))
                    steps = len(result['erewards'])
                    dummy_rew = np.zeros((steps, self.action_space))
                    dummy_old = np.zeros((steps, self.action_space))
                    dummy_ref = np.zeros((steps, self.action_space))
                    dummy_crange = np.zeros((steps, 1))
                    dummy_states = np.zeros((steps, *self.state_shape))
                    result['policy'] = self.amodel.predict([result['states'], dummy_rew, dummy_states, dummy_ref, dummy_crange])


                    for key in result.keys():
                        self.buffer[key].extend(np.copy(result[key]))
                    coords = result['coords']
                    entropy = self.get_entropy(result['policy'])
                    self.stats['x'].extend(list(coords[:-self.cutted_steps,0]))
                    self.stats['y'].extend(list(coords[:-self.cutted_steps,1]))
                    self.stats['some_map'].extend(list(result['irewards'][:-self.cutted_steps]))
                    self.stats['ireward_map'].extend(list(result['irewards'][:-self.cutted_steps]))
                    self.stats['entropy_map'].extend(list(entropy[:-self.cutted_steps]))
                    self.stats['cutted'].append(steps-self.cutted_steps)
                # for key in self.buffer.keys():
                #     print(key, len(self.buffer[key]))
                # iadvantages, itargets, ivariance = self.compute_advantages(self.buffer['states'],
                #                                                     self.buffer['irewards'],
                #                                                     self.buffer['actions'],
                #                                                     self.ivmodel,
                #                                                     self.igamma,
                #                                                     self.lam,
                #                                                     episodic=True,
                #                                                     trace=False)

                # self.buffer['iadvantages'] =iadvantages
                # self.buffer['itargets'] = itargets
                # self.buffer['ivariance'] = ivariance
                print('buffer size', getsizeof(self.buffer)/1024**2)
                for key in self.buffer.keys():
                    self.buffer[key] = np.array(self.buffer[key], dtype=self.all_keys[key])
                self.buffer['advantages'] = self.iw * self.buffer['iadvantages'] + self.ew * self.buffer['eadvantages']
                    
                

                

                usteps = math.ceil(len(self.buffer['states'])/self.batch_size)-1

                


                print('training policy model...')
                
                temp_adv = np.sum(self.buffer['advantages'], axis=-1).reshape((-1,))
                #temp_adv -= np.std(temp_adv)    
                bool_adv = temp_adv>0
                pos_len = len(bool_adv[bool_adv])
                print('pos_parts', pos_len/ len(temp_adv))
                if pos_len>0:
                    losses = []
                    ef = min(len(self.buffer['states'][bool_adv])/self.memory_size, 1)
                    mask = []
                    for i, adv in enumerate(self.buffer['advantages'][bool_adv]):
                        elem = adv/np.sum(adv, axis=-1)
                        mask.append(elem)
                        #self.buffer['advantages'][bool_adv][i] = elem 
                    mask = np.array(mask)
                    masked_policy = (self.buffer['policy'][bool_adv]) * mask
                    choosed_policy = np.sum(masked_policy, axis=-1)
                    ratios = (1-choosed_policy)/(1-1/self.action_space)
                    ratios = np.clip(ratios, 0, 1)
                    crange = np.zeros((len(self.buffer['states'][bool_adv]),1))
                    crange[:,0] += self.crange * ratios * ef
                    count = 0
                    
                    while count<20:
                        randomize = np.arange(len(self.buffer['advantages'][bool_adv]))
                        np.random.shuffle(randomize)
                        random_states = self.buffer['states'][bool_adv]
                        random_policies = self.buffer['policy'][bool_adv]
                        if count==0:
                            losses.append(self.amodel.evaluate(x=[self.buffer['states'][bool_adv], self.buffer['advantages'][bool_adv], random_states, random_policies, crange],
                                                                batch_size = self.batch_size))
                            print('evaluated loss:', losses[-1])
                        history = self.amodel.fit(x=[self.buffer['states'][bool_adv], self.buffer['advantages'][bool_adv], random_states, random_policies, crange],
                                        batch_size = self.batch_size, epochs=10)
                        losses.append(history.history['loss'][-1])
                        peak = np.argmax(losses)
                        count += 1
                        if len(losses)> peak+self.count+1:
                            diffl = np.array(losses[:-1]) - np.array(losses[1:])
                            ratio = np.mean(diffl[-self.count:])/np.mean(diffl[peak:peak+self.count])
                            # if diffl[-1]/diffl[-2] > 0.5:
                            #     self.stats['policy_lr'] = np.clip(1.2 * self.stats['policy_lr'], 1e-5, 1e-1)
                            # else:
                            #     self.stats['policy_lr'] = np.clip(0.8 * self.stats['policy_lr'], 1e-5, 1e-1)
                            # K.set_value(self.amodel.optimizer.lr, self.stats['policy_lr'])
                            print('ratio:', round(ratio, 6), 'lr', round(self.stats['policy_lr'], 6))
                            if ratio<0.3 and losses[-1]<losses[0]:
                                break
                        


                # print('training ext value model...')       
                # losses = []
                # variance = np.mean(self.buffer['evariance'][:,0])
                # # print('evariance', variance)
                # sigmas = np.zeros(self.buffer['evariance'].shape) + variance    
                # while True:
                #     history = self.evmodel.fit(x=[self.buffer['states'], self.buffer['etargets'], sigmas], 
                #                                 batch_size = self.batch_size, epochs=1)
                #     losses.append(history.history['loss'][-1])
                #         #break
                #     if len(losses)> 11:
                #         diffl = np.array(losses[:-1]) - np.array(losses[1:])
                #         ratio = np.mean(diffl[-10:])/np.mean(np.abs(diffl))
                #         # if diffl[-1]/diffl[-2] > 0.5:
                #         #     self.stats['policy_lr'] = np.clip(1.2 * self.stats['policy_lr'], 1e-5, 1e-1)
                #         # else:
                #         #     self.stats['policy_lr'] = np.clip(0.8 * self.stats['policy_lr'], 1e-5, 1e-1)
                #         # K.set_value(self.amodel.optimizer.lr, self.stats['policy_lr'])
                #         print('ratio:', round(ratio, 6), 'lr', round(self.stats['evmodel_lr'], 6))
                #         if abs(ratio)<0.2 and losses[-1]<losses[0]:
                #             break

                print('training int value model...')       
                losses = []
                preloss = np.zeros(self.buffer['itargets'].shape)
                preloss = self.ivmodel.predict(x=[self.buffer['states'], self.buffer['itargets'], preloss])
                evaluated = self.ivmodel.evaluate(x=[self.buffer['states'], self.buffer['itargets'], preloss], batch_size=self.batch_size)
                count = 0          
                while count<20:
                    if count == 0:
                        losses.append(self.ivmodel.evaluate(x=[self.buffer['states'], self.buffer['itargets'], preloss], 
                                                batch_size = self.batch_size))
                        print('evaluated loss:', losses[-1])
                    history = self.ivmodel.fit(x=[self.buffer['states'], self.buffer['itargets'], preloss], 
                                                batch_size = self.batch_size, epochs=1)
                    losses.append(history.history['loss'][-1])
                    peak = np.argmax(losses)
                    count += 1
                    if len(losses)> peak+self.count+1:
                        diffl = np.array(losses[:-1]) - np.array(losses[1:])
                        ratio = np.mean(diffl[-self.count:])/np.mean(diffl[peak:peak+self.count])
                        # if diffl[-1]/diffl[-2] > 0.5:
                        #     self.stats['policy_lr'] = np.clip(1.2 * self.stats['policy_lr'], 1e-5, 1e-1)
                        # else:
                        #     self.stats['policy_lr'] = np.clip(0.8 * self.stats['policy_lr'], 1e-5, 1e-1)
                        # K.set_value(self.amodel.optimizer.lr, self.stats['policy_lr'])
                        print('ratio:', round(ratio, 6), 'lr', round(self.stats['ivmodel_lr'], 6))
                        if ratio<0.2 and losses[-1]<losses[0]:
                            break
                
                print('training imodel...')
                states = self.reusable_memory[-1]['states']
                preloss = self.get_iloss(states)
                print('shape of preloss', preloss.shape)
                steps = len(preloss)
                if self.stats['initialized'] > 0:
                    self.stats['imean'] = ((self.horizon-steps)*self.stats['imean']+steps*np.mean(preloss))/self.horizon
                    self.stats['istd'] = ((self.horizon-steps)*self.stats['istd']+steps*np.std(preloss))/self.horizon
                    self.stats['imax'] = ((self.horizon-steps)*self.stats['imax']+steps*np.max(preloss))/self.horizon
                    self.stats['imin'] = ((self.horizon-steps)*self.stats['imin']+steps*np.min(preloss))/self.horizon
                else:
                    self.stats['imean'] = np.mean(preloss)
                    self.stats['istd'] = np.std(preloss)
                    self.stats['imax'] = np.max(preloss)
                    self.stats['imin'] = np.min(preloss)
                self.stats['initialized'] += 1
                choosed = preloss > self.stats['imean']
                stds = np.zeros((*preloss.shape, 1))
                means = np.zeros((*preloss.shape, 1))
                stds[:,0] += self.stats['istd']
                means[:,0] += self.stats['imean']
                temp = np.zeros((*preloss.shape, 1))
                temp[:,0] += preloss
                preloss = temp 
                losses = []
                # print('evaluated loss:', np.mean(ratios), 'maxr:', np.max(ratios), 'minr:', np.min(ratios), 'stds:', np.std(ratios))
                if len(choosed[choosed])>0:
                    count = 0 
                    while count<20:
                        count += 1
                        history = self.reward_model.fit(x=[states, preloss, stds, means],
                                        batch_size = 1)
                        losses.append(history.history['loss'][-1]) 
                        # peak = np.argmax(losses)
                        # if len(losses)> peak+11:
                        #     diffl = np.array(losses[:-1]) - np.array(losses[1:])
                        #     ratio = np.mean(diffl[-10:])/np.mean(diffl[peak:peak+10])
                        #     # if diffl[-1]/diffl[-2] > 0.5:
                        #     #     self.stats['policy_lr'] = np.clip(1.2 * self.stats['policy_lr'], 1e-5, 1e-1)
                        #     # else:
                        #     #     self.stats['policy_lr'] = np.clip(0.8 * self.stats['policy_lr'], 1e-5, 1e-1)
                        #     # K.set_value(self.amodel.optimizer.lr, self.stats['policy_lr'])
                        #     print('ratio:', round(ratio, 6), 'lr', round(self.stats['imodel_lr'], 6))
                        #     if abs(ratio)<0.2:
                        #         break
                        if losses[-1]<0.2:
                            break


            
                     

                # # l = []
                # # for _ in range(math.ceil(self.maxstat/2)):
                # #     r=random.randint(math.ceil(len(self.stats['episodes_numbers'])/2),
                # #                             len(self.stats['episodes_numbers'])-1)
                # #     if r not in l: 
                # #         l.append(r)

                for key in self.stats.keys():
                    if isinstance(self.stats[key], list):
                        # if (key not in excluded) and (len(self.stats[key])>self.maxstat):
                        #     new = []
                        #     for r in sorted(l):
                        #         new.append(self.stats[key][r])
                        #     self.stats[key][-len(new):] = new
                        # if key not in self.excluded and len(self.stats[key])>1000:
                        #     self.stats[key] = self.stats[key][::2]
                        self.stats[key] = list(map(lambda x: float(x), self.stats[key]))
                        
                    elif isinstance(self.stats[key], np.generic):
                        self.stats[key] = float(self.stats[key])

                lock.acquire()
                self.save_stats()
                self.save_models()
                lock.release()
                #self.reusable_memory = []


                if self.render:
                    render_que.put(True)
                
                if self.stats['mean100_external_rewards'][-1]>99500:
                    print('agent completed training successfully in '+str(self.stats['episodes_passed'])+' episodes')
                    for i in range(self.num_workers):
                        last_word = 'process '+str(processes[i].pid)+' terminated'
                        processes[i].terminate()
                        print(last_word)
                    break
            
    
    def get_stats(self, path=r'D:\sonic_models\result.json'):
        try:
            with open(path, 'r') as file:
                self.stats = json.load(file)
        except:
            print('loading default stats...')
            self.stats['steps_list'] = []
            self.stats['episodes_passed'] = 0
            self.stats['initialized'] = 0
            self.stats['episodes_numbers'] = []
            self.stats['imean'] = 0
            self.stats['istd'] = 1
            self.stats['imax'] = 1
            self.stats['imin'] = 0
            self.stats['window'] = 10
            self.stats['mean100_external_rewards'] = []
            self.stats['external_rewards'] = []
            self.stats['external_rewards_per_step']  = []
            self.stats['entropy'] = []
            self.stats['some_map'] = []
            self.stats['ireward_map'] = []
            self.stats['entropy_map'] = []
            self.stats['x'] = []
            self.stats['y'] = []
            self.stats['cutted'] = []
            self.stats['action_std'] = []


    def actor_data_generator(self, states, advantages, policies):
        steps = len(states)
        while True:
            for ind in range(0, steps-self.batch_size, self.batch_size):
                state_batch = np.array(states[ind:ind+self.batch_size,:], dtype=np.uint8)
                dummy_action_batch = np.zeros((self.batch_size,self.action_space))
                advantage_batch = np.array(advantages[ind:ind+self.batch_size,:], dtype=np.float32)
                old_batch = np.array(policies[ind:ind+self.batch_size,:], dtype=np.float32)
                yield [state_batch, advantage_batch,  old_batch], dummy_action_batch

    def critic_data_generator(self, states, values):
        steps = len(states)
        while True:
            for ind in range(0, steps-self.batch_size, self.batch_size):
                state_batch = np.array(states[ind:ind+self.batch_size,:], dtype=np.uint8)    
                value_batch = np.array(values[ind:ind+self.batch_size], dtype=np.float32)
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

    def imodel_generator(self, state_buffer):
        steps = len(state_buffer)
        batch_size = self.batch_size
        while True:
            for ind in range(0, steps-batch_size, batch_size):
                state_batch = np.array(state_buffer[ind:ind+batch_size,:], dtype=np.uint8)
                dummy_target = np.zeros((batch_size,1))
                yield state_batch, dummy_target
    
    def compute_advantages(self, states, rewards, actions, value_model, gamma, lam, episodic, trace):
        steps = len(states)
        states = self.process_states(states)
        dummy_sigma = np.zeros((steps, 1))
        dummy_target = np.zeros((steps, 1))
        values = value_model.predict([states, dummy_target, dummy_sigma])
        
        rewards = np.array(rewards)
        # print('reards shape', rewards.shape)
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
        
        advantages = np.zeros((steps,self.action_space)).astype(self.adv)
        target_values = np.zeros((steps,1)).astype(self.adv)
        td = np.zeros((steps,1)).astype(self.adv)
        rewards = rewards.reshape((-1,1))
        
        # if self.stats['initialized']<1000000:
        # target_values[:,0] = G[:]
        # else:
        # print('rrr', rewards.shape)
        # print('vvvv', values.shape)
        # print('tttt', target_values.shape)
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
        return advantages, target_values, td ** 2

    def run_episode(self, Amodel, render=False, record=False, game_id=0, path='.'):
        import retro
        import random
        if record:
            env = retro.make(game=self.game_name, state=self.map_name, scenario=self.scenario, record=path)    
        else:
            env = retro.make(game=self.game_name, state=self.map_name, scenario=self.scenario)
        env.movie_id = game_id
        cur_mem = deque(maxlen = self.state_len)
        done = False
        delay_limit = self.timedelay
        frame_reward = 0
        states = []
        erewards = []
        actions = []
        policy = []
        coords = []
        for _ in range(self.state_len):
            cur_mem.append(np.zeros((self.height, self.width, 3),dtype=self.default))
        next_state = env.reset()
        while not done:
            next_state = self.resize_state(next_state)
            cur_mem.append(next_state)
            states.append(np.array(cur_mem, dtype=np.uint8))
            action_id, cur_policy = self.choose_action(cur_mem, Amodel, render)
            policy.append(cur_policy)
            actions.append(action_id)
            for _ in range(random.randint(delay_limit-0, delay_limit+0)):
                next_state, reward, done, info = env.step(self.actions[action_id])
                frame_reward += reward
            erewards.append(frame_reward)
            coords.append([info['x'], info['y']])
            frame_reward = 0
            if render:
                env.render()
        env.render(mode='rgb_array', close=True)
        return {'states':states,
                'erewards':erewards,
                'actions': actions,
                'policy': policy,
                'coords': coords}
            

    def worker(self, l, data, render_que, game_ids, use_gpu=True):
        import os
        import time

        if not use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        l.acquire()
        amodel = policy_net(self.state_shape, self.action_space)
        if os.path.isfile(r'D:\sonic_models\policy_model.h5'):
            amodel.load_weights(r'D:\sonic_models\policy_model.h5')
            policy_loading_time = os.path.getmtime(r'D:\sonic_models\policy_model.h5')
        else:
            policy_loading_time=0
        l.release()

        while True:
            l.acquire()
            if os.path.isfile(r'D:\sonic_models\policy_model.h5'):
                if policy_loading_time<os.path.getmtime(r'D:\sonic_models\policy_model.h5'):
                    policy_loading_time = os.path.getmtime(r'D:\sonic_models\policy_model.h5')
                    try:
                        amodel.load_weights(r'D:\sonic_models\policy_model.h5')
                        print('new weights are loaded by process', os.getpid())
                    except:
                        print('new weights are not loaded by process', os.getpid())
            l.release()
            game_id = game_ids.get()
            render = False
            if render_que.qsize()>0:
                render = render_que.get()
            # if game_id in self.special_game_ids:
            #     record_replay=True
            # else:
            record_replay=False
            result = self.run_episode(amodel, render=render, record=record_replay, game_id=game_id, path=r'D:\sonic_models\replays')
            while True:
                if data.qsize()<1:
                    data.put(result)
                    break
                else:
                    time.sleep(10)

    def choose_action(self, state_col, Amodel, render):
        states = np.reshape(np.array(state_col, dtype=np.uint8), (1, *self.state_shape))
        dummy_rew = np.zeros((1, self.action_space))
        dummy_old = np.zeros((1, self.action_space))
        dummy_ref = np.zeros((1, self.action_space))
        dummy_crange = np.zeros((1, 1))
        dummy_states = np.zeros((1, *self.state_shape))
        policy = Amodel.predict([states, dummy_rew, dummy_states, dummy_ref, dummy_crange])
        policy = policy[-1]
        # if render:
        #     print([round(pol,6) for pol in policy])
        order = np.random.choice(self.action_space, size=None, p=policy)
        return order, policy

    def process_states(self, states):
        states = np.array(states, dtype=np.uint8)
        states.reshape((len(states), *self.state_shape))
        return states.astype(np.uint8)

    def get_actions(self):
        target = []
        buttons = ('B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z')
        #actions = [[], ['LEFT'], ['RIGHT'], ['LEFT','DOWN'], ['RIGHT','DOWN'], ['DOWN'], ['DOWN', 'B'], ['B'], ['LEFT','B'], ['RIGHT','B']]
        actions = [['LEFT'], ['RIGHT'], ['LEFT','B'], ['RIGHT','B'], ['B'], ['DOWN']]
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

    def update_stats(self, result):
        erewards = result['erewards']
        irewards = result['irewards']
        policy = result['policy']
        entropy = self.get_entropy(policy)
        steps = len(erewards)
        self.stats['action_std'].append(np.std(policy))
        self.stats['episodes_passed'] += 1
        self.stats['episodes_numbers'].append(self.stats['episodes_passed'])
        self.stats['steps_list'].append(steps) 
        self.stats['entropy'].append(np.mean(entropy))
        self.stats['external_rewards'].append(np.sum(erewards))
        self.stats['external_rewards_per_step'].append(np.sum(erewards)/steps)
        self.stats['mean100_external_rewards'].append(np.mean(self.stats['external_rewards'][-100:]))
        

        

        print('episode: ', self.stats['episodes_passed'],
            'episode_reward: ', self.stats['external_rewards'][-1],
            'mean_reward: ', self.stats['mean100_external_rewards'][-1])
        if len(self.stats['entropy'])>1:
            print('mean_entropy:', (self.stats['entropy'][-1]*steps+((2*self.memory_size)-steps)*self.stats['entropy'][-2])/(2*self.memory_size))
        else:
            print('mean_entropy:', self.stats['entropy'][-1])
        print('sum_irewards:', np.sum(irewards))
        print('steps:', steps)
        print('std:', np.std(policy, axis=0))
    
    def reinitialize_buffer(self):
        
        self.buffer=dict()
        for key in self.all_keys:
            self.buffer[key]=list()    
    
    def get_memory_length(self):
        memory_length = 0
        for result in self.reusable_memory:
            memory_length += len(result['states'])
        return memory_length

    def reduce_memory(self):
        while self.get_memory_length()>self.memory_size and len(self.reusable_memory)>2:
            # index = np.argmin(self.mean_adv)
            # self.reusable_memory.pop(index)
            # self.mean_adv.pop(index)
            self.reusable_memory.pop(0)
        # while len(self.reusable_memory)>1:
        #     self.reusable_memory.pop(0)
        gc.collect()


    def clear_maps(self):
        self.excluded = ('x', 'y', 'some_map', 'ireward_map', 'entropy_map', 'cutted')
        for elem in self.excluded:
            self.stats[elem] = []


    def save_models(self):
        self.amodel.save_weights(r'D:\sonic_models\policy_model.h5')
        self.evmodel.save_weights(r'D:\sonic_models\evalue_model.h5')
        self.ivmodel.save_weights(r'D:\sonic_models\ivalue_model.h5')
        self.reward_model.save_weights(r'D:\sonic_models\reward_model.h5')

    
    def save_stats(self, path=r'D:\sonic_models\result.json'):
        try:
            with open(path, 'w') as file:
                json.dump(self.stats, file, indent=4)
            print('Stats saved')
        except:
            print('Error occured for saving stats')
    

    def create_models(self):

        self.amodel = policy_net(self.state_shape, self.action_space)
        K.set_value(self.amodel.optimizer.lr, self.stats['policy_lr'])
        if os.path.isfile(r'D:\sonic_models\policy_model.h5'):
            self.amodel.load_weights(r'D:\sonic_models\policy_model.h5')
        

        self.evmodel = critic_net(self.state_shape, self.epsilon)
        weights = [np.array(w) for w in self.evmodel.get_weights()]
        weights[-1] *= 0
        self.evmodel.set_weights(weights)
        K.set_value(self.evmodel.optimizer.lr, self.stats['evmodel_lr'])
        if os.path.isfile(r'D:\sonic_models\evalue_model.h5'):
            self.evmodel.load_weights(r'D:\sonic_models\evalue_model.h5')

        self.ivmodel = critic_net(self.state_shape, self.epsilon)
        weights = [np.array(w) for w in self.ivmodel.get_weights()]
        weights[-1] *= 0
        self.ivmodel.set_weights(weights)
        K.set_value(self.ivmodel.optimizer.lr, self.stats['ivmodel_lr'])
        if os.path.isfile(r'D:\sonic_models\ivalue_model.h5'):
            self.ivmodel.load_weights(r'D:\sonic_models\ivalue_model.h5')

        self.reward_model = reward_net(self.state_shape)
        K.set_value(self.reward_model.optimizer.lr, self.stats['imodel_lr'])
        if os.path.isfile(r'D:\sonic_models\reward_model.h5'):
            self.reward_model.load_weights(r'D:\sonic_models\reward_model.h5')
    
    def get_iloss(self, states):
        dummy_old = np.zeros((len(states),))
        dummy_std = np.zeros((len(states),))
        dummy_mean = np.zeros((len(states),))
        return self.reward_model.predict(x=[states, dummy_old, dummy_std, dummy_mean])


    def run_workers(self, num_workers):
        pass   
               



if __name__ == '__main__':
    agent = SonicAgent()
    agent.run_train()
    # result_file = open(r'D:\sonic_models\result.json', 'r')
    # result = json.load(result_file)
    # episode_passed = result['episode_passed']
    # Amodel = create_policy_net('policy', 1)
    # if os.path.isfile(r'D:\sonic_models\policy_model.h5'):
    #     Amodel.load_weights(r'D:\sonic_models\policy_model.h5')
    # agent.run_episode(Amodel, render=True, record=True, game_id=episode_passed, path=r'D:\sonic_models\replays')


