# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 00:37:57 2020

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


#Data Retrieval
dataset_training = pd.read_csv('Dataset\\Training\\Mirai_dataset.csv')
print(dataset_training.head())
labels_training = pd.read_csv('Dataset\\Training\\mirai_labels.csv')

dataset_testing = pd.read_csv('Dataset\\Testing\\mirai_testing_dataset.csv')
labels_testing = pd.read_csv('Dataset\\Testing\\mirai_testing_labels.csv')



#Getting X and Y Data
X_train = dataset_training.iloc[:, 1:]
Y_train = labels_training.iloc[:, :]

X_test = dataset_testing.iloc[:, :-1]
Y_test = labels_testing.iloc[:, :]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

test = np.array(X_train)

test = np.array(X_train)

X_train = tf.reshape(X_train, (764137, 115, 1))
X_test = tf.reshape(X_test, (534388, 115, 1))


##ACTUAL GAN TRAINING

EPOCHS = 50
SEED_SIZE = 100
BATCH_SIZE = 16


#Models Load Up Load-Up
json_file = open('pretrain_disc.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

discriminator = tf.keras.models.model_from_json(loaded_model_json)
discriminator.load_weights("pretrain_disc.h5")

preds_test = model.predict_classes(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy = {}".format(accuracy_score(Y_test, preds_test)))

#Generator

generator = build_generator(SEED_SIZE)



generator_optimizer = tf.keras.optimizers.Adam(1.5e-4,0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1.5e-4,0.5)

# GAN Training

def train_step(dataset):
    seed = tf.random.normal([BATCH_SIZE,SEED_SIZE])
    print(seed)
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_dataset = generator(seed)
        
        real_output = discriminator(images)
        fake_output = discriminator(generated_dataset)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        
        
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        return gen_loss,disc_loss
    

def train(dataset, epochs):
  fixed_seed = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, SEED_SIZE))
  start = time.time()

  for epoch in range(epochs):
    epoch_start = time.time()

    gen_loss_list = []
    disc_loss_list = []

    for image_batch in dataset:
      t = train_step(image_batch)
      gen_loss_list.append(t[0])
      disc_loss_list.append(t[1])

    g_loss = sum(gen_loss_list) / len(gen_loss_list)
    d_loss = sum(disc_loss_list) / len(disc_loss_list)

    epoch_elapsed = time.time()-epoch_start
    print (f'Epoch {epoch+1}, gen loss={g_loss},disc loss={d_loss},'\
           ' {hms_string(epoch_elapsed)}')

  elapsed = time.time()-start
  print (f'Training time: {hms_string(elapsed)}')
  

generator = build_generator(100)

noise = tf.random.normal([1,100])
generated_image = generator(noise, training=False)