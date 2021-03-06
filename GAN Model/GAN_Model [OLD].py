# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 18:26:10 2020

@author: Gavin
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense 
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv1D, Conv2DTranspose, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os 
import time
import matplotlib.pyplot as plt
tf.test.gpu_device_name()

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

##CNN
# model = Sequential()
# model.add(Conv1D(filters= 64, kernel_size=3, activation ='relu',strides = 2, padding = 'valid', input_shape= (1000, 1)))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(filters= 128, kernel_size=3, activation ='relu',strides = 2, padding = 'valid'))
# model.add(MaxPooling1D(pool_size=2))

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2DTranspose, Lambda


def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'):
    """
        input_tensor: tensor, with the shape (batch_size, time_steps, dims)
        filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
        kernel_size: int, size of the convolution kernel
        strides: int, convolution step size
        padding: 'same' | 'valid'
    """
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

def build_generator(seed_size):
    model = Sequential()



    model.add(Conv1DTranspose(input_tensor = (160, seed_size, 1), filters = 256,kernel_size=3,padding="same")) #padding=same
    model.add(BatchNormalization()) #momentum=0.8
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv1DTranspose(input_tensor = (160, seed_size, 1), filters = 128, kernel_size=3, strides=2, padding="same")) #padding=same
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv1DTranspose(input_tensor = (160, seed_size, 1), filters = 64, kernel_size=3, strides=2, padding="same")) #padding=same
    model.add(BatchNormalization())
    model.add(LeakyReLUalpha=0.2())
    
    model.add(Dense(1224)) #activation linear or relu?
    
    return model
   
    # Output resolution, additional upsampling
    # model.add(Conv2D(128,kernel_size=3,padding="same")) #padding=same
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(LeakyReLU()
    
    # Final CNN layer
    # model.add(Activation("tanh"))

    

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def build_ip_encoder():
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape = ()))
    model.add(Dense())
    

def build_discriminator():
    model = Sequential()

    model.add(Conv1D(32, kernel_size=3, strides=2, input_shape = (9,1), padding="same")) #padding=same
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv1D(64, kernel_size=3, strides=2, padding="same")) #padding=same
    # model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv1D(128, kernel_size=3, strides=2, padding="same")) #padding=same
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv1D(256, kernel_size=3, strides=1, padding="same")) #padding=same
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # model.add(Activation('tanh'))

    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model

def build_disc_2(input_shape = (12, 1)):
    
    in_value = Input(shape=input_shape)
    
    # model = Model(inputs=[in_value], outputs=[output])
    # model.compile(loss='linear', optimizer=Adam(lr=0.0001, beta_1=0.5))
    
    #Main Discriminator
    d = Conv1D(32, kernel_size=3, strides=2, padding="same")([in_value])
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Dropout(0.25)(d)
    d = Conv1D(64, kernel_size=3, strides=2, padding="same")(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Dropout(0.25)(d)
    d = Conv1D(128, kernel_size=3, strides=2, padding="same")(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Dropout(0.25)(d)
    d = Conv1D(256, kernel_size=3, strides=2, padding="same")(d)
    d = BatchNormalization()(d)
    d = Activation('tanh')(d)
    
    d = GlobalAveragePooling1D()(d)
    d = Dropout(0.25)(d)
    d = Flatten()(d)
    
    #determine if real/generated
    # d_out_layer = Dense(1, activation='sigmoid')(d)
    # d_model = Model(inputs=[in_value, d_out_layer])
    # d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    
    #determine if benign/malware after determining real
    c_out_layer = Dense(1, activation='sigmoid')(d)
    c_model = Model(inputs=[in_value, c_out_layer])
    c_model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate=0.001), metrics=['accuracy']) 
    
    return c_model

def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    #create the gan model
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    
    return model
    
        
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


#plot the accuracy and the validation accuracy
def plot_acc(h):
    plt.plot(h.history['accuracy'])
    plt.plot(h.history['val_accuracy'])

    plt.title('model accuracy')
    plt.ylabel('accuracy and loss')
    plt.xlabel('epoch')

    plt.legend(['acc', 'val acc' ], loc='upper left')
    plt.show()
    
#plot the loss and validation loss
def plot_loss(h):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('accuracy and loss')
    plt.xlabel('epoch')

    plt.legend(['loss', 'val loss' ], loc='upper left')
    plt.show()

def printabc():
    global x
    print(x + 1)