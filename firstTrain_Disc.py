# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 00:20:32 2020

@author: Gavin
"""


import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense 
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import sys

sys.path.append("D:\\Repos\\MiraiGAN")
from GAN_Model import *

dataset = pd.read_csv('Mirai_dataset.csv')
print(dataset.head())
labels = pd.read_csv('mirai_labels.csv')

#Getting X and Y Data
X_train = dataset.iloc[:, 1:]
Y_train = labels.iloc[:, :]

print(X.shape)

X_train = tf.reshape(X_train, (7764137, 115, 1))

disc = build_discriminator()
disc.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate=0.001), metrics=['accuracy']) #default Adam optimizer

#Training the Discriminator on datset
history = model.fit(X_train, Y_train, epochs = 25, batch_size = 100, validation_data = (X_test, Y_test), shuffle = True)

#Saving disc model

#Metrics?
