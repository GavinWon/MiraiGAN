# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 19:25:39 2020

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
from GAN_Model2 import *





test = np.array(X_train)

test = np.array(X_train)
print(np.mean(test))
print(np.amin(test))
print(np.amax(test))


print(X.shape)

X_train = tf.reshape(X_train, (79794, 1224, 1))
X_test = tf.reshape(X_test, (26598, 1224, 1))

disc = build_discriminator()

#Training the Discriminator on datset
history = disc.fit(X_train, Y_train, epochs = 10, batch_size = 100, validation_data = (X_test, Y_test), shuffle = True)

#Saving disc model
file_name = "pretrain_disc"
    
json = file_name + ".json"
h5 = file_name + ".h5"
    
model_json = disc.to_json()
with open(json, "w") as json_file:
    json_file.write(model_json)
    disc.save_weights(h5)

#Metrics

from sklearn.metrics import accuracy_score
preds_training = disc.predict_classes(X_train)
preds_test = disc.predict_classes(X_test)
print("Accuracy = {}".format(accuracy_score(Y_test, preds_test)))
print("Accuracy = {}".format(accuracy_score(Y_train, preds_training)))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, preds_test))
print(confusion_matrix(Y_train, preds_training))
plot_acc(history)
plot_loss(history)

print(disc.summary())
