# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 21:01:02 2020

@author: Gavin
"""
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


from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
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


'''                         GAN STRURCTURE                                 '''

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

def custom_activation(output):
	logexpsum = np.sum(np.exp(output))
	result = logexpsum / (logexpsum + 1.0)
	return result

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
    
    model.add(Dense(9)) #activation linear or relu?
    
    return model
   
    # Output resolution, additional upsampling
    # model.add(Conv2D(128,kernel_size=3,padding="same")) #padding=same
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(LeakyReLU()
    
    # Final CNN layer
    # model.add(Activation("tanh"))

def build_disc_ip():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape = ()))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(20, activation="softmax")) #output layer
    
    return model

def build_main_disc():
    #determine if real/generated
    d_model = Sequential()

    d_model.add(Conv1D(32, kernel_size=3, strides=2, input_shape= (30, 1), padding="same")) #padding=same
    d_model.add(LeakyReLU(alpha=0.2))

    d_model.add(Dropout(0.25))
    d_model.add(Conv1D(64, kernel_size=3, strides=2, padding="same")) #padding=same
    # model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    d_model.add(BatchNormalization())
    d_model.add(LeakyReLU(alpha=0.2))

    d_model.add(Dropout(0.25))
    d_model.add(Conv1D(128, kernel_size=3, strides=2, padding="same")) #padding=same
    d_model.add(BatchNormalization())
    d_model.add(LeakyReLU(alpha=0.2))

    d_model.add(Dropout(0.25))
    d_model.add(Conv1D(256, kernel_size=3, strides=1, padding="same")) #padding=same
    d_model.add(BatchNormalization())
    d_model.add(Activation('tanh'))

    d_model.add(GlobalAveragePooling1D())
    d_model.add(Dropout(0.25))
    d_model.add(Flatten())
    
    
    d_model.add(Dense(1, activation='sigmoid')) #d_out_layer = Lambda(custom_activation)(fe)
    d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    
    #determine if benign/malware after determining real
    c_model = Sequential()

    c_model.add(Conv1D(32, kernel_size=3, strides=2, input_shape= (30, 1), padding="same")) #padding=same
    c_model.add(LeakyReLU(alpha=0.2))

    c_model.add(Dropout(0.25))
    c_model.add(Conv1D(64, kernel_size=3, strides=2, padding="same")) #padding=same
    # model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    c_model.add(BatchNormalization())
    c_model.add(LeakyReLU(alpha=0.2))

    c_model.add(Dropout(0.25))
    c_model.add(Conv1D(128, kernel_size=3, strides=2, padding="same")) #padding=same
    c_model.add(BatchNormalization())
    c_model.add(LeakyReLU(alpha=0.2))

    c_model.add(Dropout(0.25))
    c_model.add(Conv1D(256, kernel_size=3, strides=1, padding="same")) #padding=same
    c_model.add(BatchNormalization())
    c_model.add(Activation('tanh'))

    c_model.add(GlobalAveragePooling1D())
    c_model.add(Dropout(0.25))
    c_model.add(Flatten())
    
    
    c_model.add(Dense(1, activation='sigmoid'))
    c_model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate=0.001), metrics=['accuracy']) #default Adam optimizer

    return d_model, c_model

# define the combined generator and discriminator model, for updating the generator
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

'''           GETTING AND GENERATING REAL/FAKE SAMPLES                    '''
# select a supervised subset of the dataset, ensures classes are balanced
def get_real_samples(dataset, n_samples=100, n_classes=2):
	# split into images and labels
	X, y = dataset
	# choose random instances
	ix = randint(0, X.shape[0], n_samples)
	# select images and labels
	X_sample, y_sample = X[ix], y[ix]
	# generate class labels
	y_other = ones((n_samples, 1))
	return [X, y], y_other  

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	z_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = z_input.reshape(n_samples, latent_dim)
	return z_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	z_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	x = generator.predict(z_input)
	# create class labels
	y = zeros((n_samples, 1))
	return x, y

'''                          LOSS FUNCTION                                '''
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

'''                          METRICS                                 '''

from matplotlib import pyplot
# generate samples and save as a plot and save the model
# NEED TO EDIT
def summarize_performance(step, g_model, c_model, latent_dim, dataset, n_samples=100):
	# prepare fake examples
	X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
	# scale from [-1,1] to [0,1]
	X = (X + 1) / 2.0
	# plot images
	for i in range(100):
		# define subplot
		pyplot.subplot(10, 10, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
	# save plot to file
	filename1 = 'generated_plot_%04d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	# evaluate the classifier model
	X, y = dataset
	_, acc = c_model.evaluate(X, y, verbose=0)
	print('Classifier Accuracy: %.3f%%' % (acc * 100))
	# save the generator model
	filename2 = 'g_model_%04d.h5' % (step+1)
	g_model.save(filename2)
	# save the classifier model
	filename3 = 'c_model_%04d.h5' % (step+1)
	c_model.save(filename3)
	print('>Saved: %s, %s, and %s' % (filename1, filename2, filename3))


'''                         ACCURACY/LOSS PLOT                                '''
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

