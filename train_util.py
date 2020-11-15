import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.models import clone_model
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
from tqdm import trange


def evaluate(inputs, model, loss_fn, batch_size=32, verbose=True, **kwargs):
    all_len = 0 
    for input in inputs:
        if hasattr(input, "__len__"):
            all_len = max(len(input), all_len)
    nsteps = all_len // batch_size
    if all_len % batch_size:
        nsteps += 1
    accumulated_loss = 0
    pbar = trange(nsteps, ncols=150, disable=not verbose)
    for step in pbar:
        start = batch_size * step
        end = min(start + batch_size, all_len)
        delta = end-start
        batch_input = [input[start:end] for input in inputs]
        with tf.GradientTape() as tape:
            batch_loss = loss_fn(batch_input, model, **kwargs)
        accumulated_loss += batch_loss.numpy()
        mean_loss = accumulated_loss/end
        pbar.set_description('Evaluating {} '.format(model.name))
        pbar.set_postfix_str('mean_loss: {:.6E}'.format(mean_loss))
    return mean_loss


def train_by_batch(inputs, model, loss_fn, optimizer=None, batch_size=32, epochs=1, verbose=True, **kwargs):
    loss_history = []
    
    if not optimizer:
        optimizer = model.optimizer
    for epoch in range(epochs):
        all_len = 0 
        for input in inputs:
            if hasattr(input, "__len__"):
                all_len = max(len(input), all_len)
        epoch_nsteps = all_len // batch_size
        if all_len % batch_size:
            epoch_nsteps += 1
        accumulated_epoch_loss = 0
        pbar = trange(epoch_nsteps, ncols=150, disable=not verbose)
        for step in pbar:
            start = batch_size * step
            end = min(start + batch_size, all_len)
            delta = end-start
            batch_input = [input[start:end] for input in inputs]
            loss = batch_train_step(batch_input, model, loss_fn, optimizer, **kwargs) 
            accumulated_epoch_loss += loss.numpy()
            mean_epoch_loss = accumulated_epoch_loss/end
            pbar.set_description('Training {}, epoch {}/{}'.format(model.name, epoch+1, epochs))
            pbar.set_postfix_str('mean_loss: {:.6E}'.format(mean_epoch_loss))
        loss_history.append(accumulated_epoch_loss / all_len)
    model.compile(optimizer=optimizer)
    return loss_history

def train_by_epoch(inputs, model, loss_fn, optimizer=None, batch_size=32, epochs=1, verbose=True, **kwargs):
    loss_history = []
    training_weights = model.trainable_variables
    if not optimizer:
        optimizer = model.optimizer
    for epoch in range(epochs):
        all_len = 0 
        for input in inputs:
            if hasattr(input, "__len__"):
                all_len = max(len(input), all_len)
        epoch_nsteps = all_len // batch_size
        if all_len % batch_size:
            epoch_nsteps += 1
        accumulated_epoch_loss = 0
        accumulated_grads = [tf.zeros_like(layer) for layer in training_weights]
        pbar = trange(epoch_nsteps, ncols=150, disable=not verbose)
        for step in pbar:
            start = batch_size * step
            end = min(start + batch_size, all_len)
            delta = end-start
            batch_input = [input[start:end] for input in inputs]
            loss, grads = grad_tape(batch_input, model, loss_fn, **kwargs)
            accumulated_epoch_loss += loss.numpy()
            mean_epoch_loss = accumulated_epoch_loss/end
            for i, grad in enumerate(grads):
                if grad is not None:
                    accumulated_grads[i] += grad
            pbar.set_description('Training {}, epoch {}/{}'.format(model.name, epoch+1, epochs))
            pbar.set_postfix_str('mean_loss: {:.6E}'.format(mean_epoch_loss))
        accumulated_grads = [gradient/all_len for gradient in accumulated_grads]
        optimizer.apply_gradients(zip(accumulated_grads, training_weights))
        loss_history.append(accumulated_epoch_loss / all_len)
    model.compile(optimizer=optimizer)
    return loss_history



def batch_train_step(inputs, model, loss_fn, optimizer, **kwargs):
    input_len = len(inputs[0])
    initial_grads = [tf.zeros_like(layer) for layer in model.trainable_variables]
    loss, grads = grad_tape(inputs, model, loss_fn, **kwargs)
    for i, grad in enumerate(grads):
        if grad is not None:
            initial_grads[i] += grad / input_len
    optimizer.apply_gradients(zip(initial_grads, model.trainable_variables))
    return loss

#@tf.function
def grad_tape(inputs, model, loss_fn, **kwargs):
    with tf.GradientTape() as tape:
        loss = loss_fn(inputs, model, **kwargs)
    grads = tape.gradient(loss, model.trainable_variables)
    return loss, grads