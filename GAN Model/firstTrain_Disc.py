"""
@author: Gavin
"""


'''import statements'''
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

'''Import GAN Model'''
sys.path.append("D:\\Repos\\MiraiGAN")
from GAN_Model2 import *



'''Reshaping the main X data'''
X_train = tf.reshape(X_train, (79794, 1214, 1))
X_test = tf.reshape(X_test, (26598, 1214, 1))

'''Reshapign the sub-datasets IP and other (non-IP)'''
X_IP_train = tf.reshape(X_IP_train, (79794, 1214, 1))
X_IP_test = tf.reshape(X_IP_test, (26598, 1214, 1))
X_other_train = tf.reshape(X_other_train, (79794, 10, 1))
X_other_test = tf.reshape(X_other_test, (26598, 10, 1))


'''Initialize the model'''
_,disc_main = build_main_disc()
disc_ip = build_ip_disc()
disc = build_discriminator(disc_ip, disc_main)

'''Training discriminator to classify benign/malware'''
history = disc.fit([X_IP_train, X_other_train], Y_train, epochs = 5, batch_size = 100, validation_data = ([X_IP_test, X_other_test], Y_test), shuffle = True)




'''Saving converted data file for GAN Training'''
other = X_other_train
IP = disc_ip.predict(X_IP_train)
final = np.hstack((IP, other)) #1-20 --> IP, #21 - 30 --> other
np.save('data.npy', final)

'''Testing discriminator prediction'''
test = X_IP_train[0]
test = tf.reshape(test, (1, 1214, 1))
predict = disc_ip.predict([test])


'''Saving discriminator model'''
file_name = "pretrain_disc"
json = file_name + ".json"
h5 = file_name + ".h5"  
model_json = disc.to_json()
with open(json, "w") as json_file:
    json_file.write(model_json)
    disc.save_weights(h5)

'''Metrics'''
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

'''Model summary'''
print(disc.summary())
