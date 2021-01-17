# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 18:35:43 2020

@author: Gavin
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 00:37:57 2020

@author: Gavin
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense 
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam

import numpy as np
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randn
from numpy.random import randint


import pandas as pd
import sys

from matplotlib import pyplot
from tensorflow.keras import backend

sys.path.append("D:\\Repos\\MiraiGAN")
from GAN_Model2 import *


# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=10, n_batch=100):

    # calculate the number of batches per training epoch
	bat_per_epo = int(X.shape[0] / n_batch)
    
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# calculate the size of half a batch of samples
	half_batch = int(n_batch / 2)
	#print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (n_epochs, n_batch, half_batch, bat_per_epo, n_steps))
	# manually enumerate epochs
	for i in range(n_steps):
        
        # update discriminator (d)
		X_real, y_real = get_real_samples(dataset, n_samples = n_batch) #Y_real is all 1
		d_loss1 = d_model.train_on_batch(X_real, y_real)
		X_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_batch) #Y_fake is all 0
		d_loss2 = d_model.train_on_batch(X_fake, y_fake)
        
        # update generator (g)
		X_gan, y_gan = generate_latent_points(latent_dim, n_batch), ones((n_batch, 1)) #Y_gan is all 1
		g_loss = gan_model.train_on_batch(X_gan, y_gan)
        
        
		
		# summarize loss on this batch
# 		print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (i+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss))
# 		# evaluate the model performance every so often
# 		if (i+1) % (bat_per_epo * 1) == 0:
# 			summarize_performance(i, g_model, c_model, latent_dim, dataset)
    #Save model for the discriminator and generator

# size of the latent space
latent_dim = 10 #or 4

# load the discriminator model
json_file = open('pretrainV1\\pretrain_disc.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

c_model = tf.keras.models.model_from_json(loaded_model_json)
c_model.load_weights("pretrainV1\\pretrain_disc.h5")

d_model, _ = build_main_disc()

# create the generator
g_model = build_generator(latent_dim)

# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
X_train = np.load('data.npy')
data = [X_train, Y_train]
# train model
train(g_model, d_model, gan_model, data, latent_dim)


##TESTING
''' 1 '''
X_real, y_real = get_real_samples(data, 100)
X_real = tf.reshape(X_real, (100, 30, 1))
d_loss1 = d_model.train_on_batch(X_real, y_real)

''' 2 '''
X_fake, y_fake = generate_fake_samples(g_model, latent_dim, 100) #Y_fake is all 0
d_loss2 = d_model.train_on_batch(X_fake, y_fake)

''' 3 '''
X_gan, y_gan = generate_latent_points(latent_dim, 100), ones((100, 1)) #Y_gan is all 1
g_loss = gan_model.train_on_batch(X_gan, y_gan)

#EXTRA
d_loss2 = d_model.train_on_batch(X_fake, y_fake)


