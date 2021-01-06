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

X_train = tf.reshape(X_train, (79794, 1214, 1))
X_test = tf.reshape(X_test, (26598, 1214, 1))


X_IP_train = tf.reshape(X_IP_train, (79794, 1214, 1))
X_IP_test = tf.reshape(X_IP_test, (26598, 1214, 1))
X_other_train = tf.reshape(X_other_train, (79794, 10, 1))
X_other_test = tf.reshape(X_other_test, (26598, 10, 1))



_,disc_main = build_main_disc()
disc_ip = build_disc_ip()
disc = build_discriminator(disc_ip, disc_main)



history = disc.fit([X_IP_train, X_other_train], Y_train, epochs = 5, batch_size = 100, validation_data = ([X_IP_test, X_other_test], Y_test), shuffle = True)

test = X_IP_train[0]
test = tf.reshape(test, (1, 1214, 1))
wubdub = disc_ip.predict([test])
wubdub2 = disc_ip.predict([test])
wubdub3 = disc_ip.predict([test])
final = disc_ip.predict(X_IP_train)







#Training the Discriminator on datset
for i in range(10):
    #Training the IP Discriminator
    history = disc.fit(X_train, Y_train, epochs = 1, batch_size = 100, validation_data = (X_test, Y_test), shuffle = True)
    
    #Training the Main Discriminator
    X_real = ip_disc.predict_classes(X_train)
    history2 = disc_main.fit(X_real, Y_train, epochs = 1, batch_size = 100, shuffle = True)

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
preds_training = np.round(disc.predict([X_IP_train, X_other_train]))
preds_test = np.round(disc.predict([X_IP_test, X_other_test]))
print("Accuracy = {}".format(accuracy_score(Y_train, preds_training)))
print("Accuracy = {}".format(accuracy_score(Y_test, preds_test)))


from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, preds_test))
print(confusion_matrix(Y_train, preds_training))
plot_acc(history)
plot_loss(history)

print(disc.summary())
