import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import *
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.models import Model
import numpy as np
#from tensorflow.keras import Input


def base_net(input_shape, summary=False):
    
    activ = 'tanh'#LeakyReLU(alpha=0.3)
    def last_image(tensor):
        return tensor[:,-1,:]

    input = Input(shape=input_shape)
    float_input = K.cast(input, dtype='float32')
    float_input = Lambda(lambda input: input/255.0-0.5)(float_input)
    float_input = Lambda(last_image)(float_input)
    x = Conv2D(48, (6,6), activation='tanh')(float_input)
    x = AveragePooling2D(pool_size=(3, 3), strides=None, padding='valid')(x)
    x = Conv2D(96, (4,4), activation='tanh')(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid')(x)
    x = Conv2D(96, (4,4), activation='tanh')(x)
    #x = TimeDistributed(activ)(x)
    # x = Conv2D(128, (2,2), strides=(1,1), padding='same')(x)
    # x = activ(x)
    x = Flatten()(x)
    output = Dense(512, activation='tanh')(x)
    #output = TimeDistributed(Dense(512, activation='tanh'))(x)
    model = Model(inputs=input, outputs=output)
    adam = SGD(lr= 1e-10, momentum=0.0)
    model.compile(optimizer=adam, loss='mean_squared_error')
    if summary:
        model.summary()
    return model

def reward_net(input_shape, summary=False):

    def custom_loss(reward, old_loss, std, means):
        r = 0.99
        loss = (reward/ old_loss - r) ** 2 / (1-r)**2
        #loss = K.clip(loss, 0.95, 1000000000.000)
        return loss

    def last_image(tensor):
        return tensor[:,-1,:]

    def ireward(x):
        return K.mean(K.square(x[0] - x[1]), axis=-1)

    input = Input(shape=input_shape)
    old_loss = Input(shape=(1,))
    std_input = Input(shape=(1,))
    mean_input = Input(shape=(1,))
    trainable_branch = base_net(input_shape)
    stochastic_branch = base_net(input_shape)
    trainable_output = trainable_branch(input)
    stochastic_output = stochastic_branch(input)
    trainable_output = Dense(512, activation='tanh')(trainable_output)
    stochastic_output = Dense(512, activation='tanh')(stochastic_output)
    # trainable_output = Lambda(last_image)(trainable_branch(input))
    # stochastic_output = Lambda(last_image)(stochastic_branch(input))
    trainable_part = Model(inputs=input, outputs=trainable_output)
    stochastic_part = Model(inputs=input, outputs=stochastic_output)
    for layer in stochastic_part.layers:
        layer.trainable = False

    
    
    intrinsic_reward = Lambda(ireward)([stochastic_output, trainable_output])
    model = Model(inputs=[input, old_loss, std_input, mean_input], outputs=intrinsic_reward)
    #optimizer = SGD(lr= 1.0e-10, momentum=0.0)
    optimizer = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-7, clipnorm=1.0)
    model.add_loss(custom_loss(intrinsic_reward, old_loss, std_input, mean_input))
    model.compile(optimizer=optimizer)
    
    if summary:
        model.summary()
    return model

def policy_net(input_shape, action_space, summary=False):

    def custom_loss(y_pred, reward_input, reference, old_reference, crange):
        xmin = 0.99
        beta= 1/2/crange
        #entropy = -K.sum(y_pred * K.log(y_pred), axis=-1) / K.log(tf.constant(action_space, tf.float32))
        # #ratio = (y_pred+1e-2)/(old_input+1e-2)
        base_loss = K.sum(-reward_input * y_pred, axis=-1)
        inertia_loss = beta *  K.sum(K.abs(reward_input), axis=-1) * K.sum(K.pow(reference-old_reference, 2), axis=-1)/tf.constant(action_space, tf.float32)
        #entropy_loss = 0.06 * beta * K.sum(K.abs(reward_input), axis=-1) * K.sum(y_pred ** 4, axis=-1)/tf.constant(action_space, tf.float32)
        # loss0 = K.sum(-reward_input * old_input + beta * K.abs(reward_input) * K.pow(old_input, 2), axis=-1)
        # ymin = K.sum(-reward_input * xmin + beta * K.abs(reward_input) * xmin ** 2, axis=-1)
        # minl = K.maximum(ymin, loss0+dnl)
        # maxl = loss0 + dpl
        # loss = K.clip(loss, minl, maxl)
        # entropy_old = -K.sum(old_input * K.log(old_input), axis=-1) / K.log(tf.constant(action_space, tf.float32))
        # d = K.pow(entropy_old, 1) * crange
        # d = tf.reshape(d, [tf.shape(d)[0],1])
        # d = crange
        # pg_loss2 =-reward_input * K.clip(ratio, 1-d, 1+d)
        # pg_loss = K.maximum(pg_loss1,pg_loss2)
        loss = base_loss + inertia_loss #+ entropy_loss 
        return loss


    # class temp_layer(Layer):
    #     def __init__(self, input_shape, action_space):
    #         super(temp_layer, self).__init__()
    #         self.xmodel = base_net(input_shape)
    #         self.space = action_space
    #     def call(self, inputs):  # Defines the computation from inputs to outputs
    #         x = self.xmodel(inputs)
    #         x = Dense(self.space, activation='softmax')(x)
    #         return x
    print('input shape is', input_shape)
    crange_input = Input(shape=(1,))
    reward_input = Input(shape=(action_space,))
    old_reference = Input(shape=(action_space,))

    old_states = Input(shape=input_shape)
    state_input = Input(shape=input_shape)
    
    conv_part = base_net(input_shape, action_space)

    main_output = Dense(action_space, activation='softmax')(conv_part(state_input))
    tmodel = Model(inputs=state_input, outputs=main_output)
    reference_output = tmodel(old_states)
    # print(conv_part.get_output_at(0))
    # print(conv_part.get_output_at(1))
    # x = LSTM(64, return_sequences=False)(x)
    # x = Dense(64, activation='linear')(x)
    
    
    model = Model(inputs=[state_input,reward_input, old_states, old_reference, crange_input], outputs=main_output)
    
    optimizer = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-7, clipnorm=1.0)
    #optimizer = NormalizedOptimizer(optimizer, normalization='max')
    #optimizer = SGD(lr= 1e-10, momentum=0.9, clipnorm=1.0)
    #optimizer = SGD(lr= 1e-5, momentum=0.0)
    model.add_loss(custom_loss(main_output, reward_input, reference_output, old_reference, crange_input))
    model.compile(optimizer=optimizer)
    if summary:
        model.summary()
    return model

def critic_net(input_shape, epsilon, summary=False):

    def custom_loss(y_pred, y_true, sigma_input):
        loss = K.square(y_pred - y_true)#/(K.square(y_true - sigma_input)+1e-6) 
        #loss = K.clip(loss0, loss0-epsilon*sigma_input, loss0+epsilon*sigma_input)
        return loss

    target_input = Input(shape=(1,))  
    sigma_input = Input(shape=(1,))
    state_input = Input(shape=input_shape)
    conv_part = base_net(input_shape)
    x = conv_part(state_input)
    # x = GRU(, return_sequences=False)(x)   
    x = Dense(512, activation='tanh')(x)
    critic_output = Dense(1, activation='linear', kernel_initializer=tf.keras.initializers.Zeros(), name='critic_output')(x)
    model = Model(inputs=[state_input, target_input, sigma_input], outputs=critic_output)
    
    optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-5, clipnorm=1.0)
    #optimizer = SGD(lr= 1e-10, momentum=0.0, clipnorm=1.0) 
    model.add_loss(custom_loss(critic_output, target_input, sigma_input))
    model.compile(optimizer=optimizer)
    if summary:
        model.summary()
    return model

# if __name__ == "__main__":
#     import numpy as np
#     width = 120
#     height = 84
#     state_shape = (1,height,width,3)
#     net = reward_net(state_shape, 10)
#     states = np.random.rand(1000,*state_shape)
    
#     preloss = net.predict()
#     steps = len(preloss)
#     if self.stats['initialized'] > 0:
#         self.stats['imean'] = ((30000-steps)*self.stats['imean']+steps*np.mean(preloss))/30000
#         self.stats['istd'] = ((30000-steps)*self.stats['istd']+steps*np.std(preloss))/30000
#         self.stats['imax'] = ((30000-steps)*self.stats['imax']+steps*np.max(preloss))/30000
#         self.stats['imin'] = ((30000-steps)*self.stats['imin']+steps*np.min(preloss))/30000
#     else:
#         self.stats['imean'] = np.mean(preloss)
#         self.stats['istd'] = np.std(preloss)
#         self.stats['imax'] = np.max(preloss)
#         self.stats['imin'] = np.min(preloss)
#     self.stats['initialized'] += 1
#     choosed = preloss > np.mean(preloss)
#     stds = np.zeros((steps,1)) + self.stats['istd']
#     temp = np.zeros((steps,1))
#     for i in range(steps):
#         temp[i] = temp[i] + preloss[i]
#     preloss = temp
#     rr = np.mean(preloss)
#     losses = []
#     print('preloss', preloss)
#     print('stds', stds)
#     print('evaluated loss:', self.reward_model.evaluate(x=[states, preloss, stds], batch_size=1))
#     print('preloss:', rr)



#     policy.fit(x=[states, rewards, astates, reference, cranges], epochs=10, batch_size=1)
    