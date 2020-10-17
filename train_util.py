import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.models import clone_model
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
from tqdm import trange

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)





def train_by_batch(inputs, model, loss_fn, optimizer, batch_size, epochs):
    loss_history = []
    training_weights = model.trainable_variables
    
    for epoch in range(epochs):
        all_len = 0 
        for input in inputs:
            all_len = max(len(input), all_len)
        epoch_nsteps = all_len // batch_size
        if all_len % batch_size:
            epoch_nsteps += 1
        accumulated_epoch_loss = 0
        pbar = trange(epoch_nsteps, ncols=200)
        for step in pbar:
            start = batch_size * step
            end = min(start + batch_size, all_len)
            delta = end-start
            accumulated_grads = [tf.zeros_like(layer) for layer in training_weights]
            batch_input = [input[start:end] for input in inputs]
            with tf.GradientTape() as tape:
                loss = loss_fn(batch_input, model)
            accumulated_epoch_loss += loss.numpy()
            mean_epoch_loss = accumulated_epoch_loss/end
            grads = tape.gradient(loss, training_weights)
            accumulated_grads = [accumulated_grads[i]+grad/delta for i, grad in enumerate(grads)]
            optimizer.apply_gradients(zip(accumulated_grads, training_weights))
            pbar.set_description('Training {}, epoch {}/{}'.format(model.name, epoch+1, epochs))
            pbar.set_postfix_str('mean_loss: {}'.format(format(mean_epoch_loss, "3.5E")))
        loss_history.append(accumulated_epoch_loss / all_len)
    return loss_history

def train_by_epoch(inputs, model, loss_fn, optimizer, batch_size, epochs):
    loss_history = []
    training_weights = model.trainable_variables
    
    for epoch in range(epochs):
        all_len = 0 
        for input in inputs:
            all_len = max(len(input), all_len)
        epoch_nsteps = all_len // batch_size
        if all_len % batch_size:
            epoch_nsteps += 1
        accumulated_epoch_loss = 0
        accumulated_grads = [tf.zeros_like(layer) for layer in training_weights]
        pbar = trange(epoch_nsteps, ncols=200)
        for step in pbar:
            start = batch_size * step
            end = min(start + batch_size, all_len)
            delta = end-start
            batch_input = [input[start:end] for input in inputs]
            
            with tf.GradientTape() as tape:
                loss = loss_fn(batch_input, model)

            accumulated_epoch_loss += loss.numpy()
            mean_epoch_loss = accumulated_epoch_loss/end
            grads = tape.gradient(loss, training_weights)
            accumulated_grads = [accumulated_grads[i] + grad for i, grad in enumerate(grads)]
            pbar.set_description('Training {}, epoch {}/{}'.format(model.name, epoch+1, epochs))
            pbar.set_postfix_str('mean_loss: {}'.format(format(mean_epoch_loss, "3.5E")))
        accumulated_grads = [gradient/all_len for gradient in accumulated_grads]
        optimizer.apply_gradients(zip(accumulated_grads, training_weights))
        loss_history.append(accumulated_epoch_loss / all_len)
    return loss_history




