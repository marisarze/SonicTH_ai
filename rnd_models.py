import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Lambda, LeakyReLU, Conv2D, Flatten, Input, TimeDistributed, LSTM, GRU, Layer
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.models import Model
from optimizer import NormalizedOptimizer
#from tensorflow.keras import Input


def base_net(input_shape, summary=False):
    
    activ = 'tanh'#LeakyReLU(alpha=0.3)


    input = Input(shape=input_shape)
    float_input = K.cast(input, dtype='float32')
    float_input = Lambda(lambda input: input/255.0-0.5)(float_input)
    x = TimeDistributed(Conv2D(32, (8,8), activation='tanh', strides=(4,4), padding='same'))(float_input)
    #x = TimeDistributed(activ)(x)
    x = TimeDistributed(Conv2D(64, (4,4), activation='tanh', strides=(2,2), padding='same'))(x)
    #x = TimeDistributed(activ)(x)
    x = TimeDistributed(Conv2D(64, (4,4), activation='tanh', strides=(2,2), padding='same'))(x)
    #x = TimeDistributed(activ)(x)
    # x = Conv2D(128, (2,2), strides=(1,1), padding='same')(x)
    # x = activ(x)
    x = TimeDistributed(Flatten())(x)
    output = LSTM(512, return_sequences=False)(x)
    #output = TimeDistributed(Dense(512, activation='tanh'))(x)
    model = Model(inputs=input, outputs=output)
    adam = SGD(lr= 1e-4, momentum=0.0)
    model.compile(optimizer=adam, loss='mean_squared_error')
    if summary:
        model.summary()
    return model

def reward_net(input_shape, summary=True):

    #@tf.function
    def custom_loss(x1, x2):
        return K.mean(K.square(x1 - x2), axis=-1)

    def last_image(tensor):
        return tensor[:,-1,:]

    def ireward(x):
        return K.mean(K.square(x[0] - x[1]), axis=-1)

    input = Input(shape=input_shape)
    trainable_branch = base_net(input_shape)
    stochastic_branch = base_net(input_shape)
    trainable_output = trainable_branch(input)
    stochastic_output = stochastic_branch(input)
    # trainable_output = Lambda(last_image)(trainable_branch(input))
    # stochastic_output = Lambda(last_image)(stochastic_branch(input))
    trainable_part = Model(inputs=input, outputs=trainable_output)
    stochastic_part = Model(inputs=input, outputs=stochastic_output)
    for layer in stochastic_part.layers:
        layer.trainable = False

    
    
    intrinsic_reward = Lambda(ireward)([stochastic_output, trainable_output])
    print('rrrr', intrinsic_reward.shape)
    model = Model(inputs=input, outputs=intrinsic_reward)
    adam = SGD(lr= 1e-3, momentum=0.0)
    model.add_loss(custom_loss(trainable_output, stochastic_output))
    model.compile(optimizer=adam)
    
    if summary:
        model.summary()
    return model

def policy_net(input_shape, action_space, summary=True):

    def custom_loss(y_pred, reward_input, reference, old_reference, crange):
        xmin = 0.99
        beta= 1/2/crange
        #entropy = -K.sum(y_pred * K.log(y_pred), axis=-1) / K.log(tf.constant(action_space, tf.float32))
        # #ratio = (y_pred+1e-2)/(old_input+1e-2)
        loss = K.sum(-reward_input * y_pred, axis=-1) + beta[:,-1] *  K.sum(K.abs(reward_input), axis=-1) * K.sum(K.pow(reference-old_reference, 2), axis=-1)
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
        # loss = K.sum(pg_loss,axis=-1) + beta * K.abs(K.sum(reward_input,axis=-1)) * K.pow(1-entropy, 1)
        return loss


    class temp_layer(Layer):
        def __init__(self, input_shape, action_space):
            super(temp_layer, self).__init__()
            self.xmodel = base_net(input_shape)

        def call(self, inputs):  # Defines the computation from inputs to outputs
            x = self.xmodel(inputs)
            x = Dense(action_space, activation='softmax')(x)
            return x
    print('input shape is', input_shape)
    crange_input = Input(shape=(1,))
    reward_input = Input(shape=(action_space,))
    old_reference = Input(shape=(action_space,))

    old_states = Input(shape=input_shape)
    state_input = Input(shape=input_shape)
    
    conv_part = temp_layer(input_shape)
    main_output = conv_part(state_input)
    reference_output = conv_part(old_states)


    # print(conv_part.get_output_at(0))
    # print(conv_part.get_output_at(1))
    # x = LSTM(64, return_sequences=False)(x)
    # x = Dense(64, activation='linear')(x)
    
    
    model = Model(inputs=[state_input,reward_input, old_states, old_reference, crange_input], outputs=main_output)
    
    adam1 = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0, amsgrad=False)
    #adam1 = NormalizedOptimizer(adam1, normalization='max')
    #adam1 = SGD(lr= 5e-5, momentum=0.0, clipnorm=1.0)
    #adam1 = SGD(lr= 2e-4, momentum=0.0)
    model.add_loss(custom_loss(main_output, reward_input, reference_output, old_reference, crange_input))
    model.compile(optimizer=adam1)
    if summary:
        model.summary()
    return model

def critic_net(input_shape, epsilon, summary=True):

    def custom_loss(y_pred, y_true, sigma_input):
        loss0 = K.square(y_pred - y_true)
        #loss = K.clip(loss0, loss0-epsilon*sigma_input, loss0+epsilon*sigma_input)
        loss = loss0
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
    
    adam1 = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-5, decay=0.0, amsgrad=False)
    #adam1 = SGD(lr= 1e-2, momentum=0, clipnorm=1.0) 
    model.add_loss(custom_loss(critic_output, target_input, sigma_input))
    model.compile(optimizer=adam1)
    if summary:
        model.summary()
    return model

if __name__ == "__main__":
    import numpy as np
    width = 120
    height = 84
    state_shape = (4,height,width,3)
    #critic = critic_net(state_shape)
    policy = policy_net(state_shape, 10)
    # reward = reward_net(state_shape)
    # A = np.random.rand(100, *state_shape)
    # S = reward.predict(A)
    # print(S.shape)