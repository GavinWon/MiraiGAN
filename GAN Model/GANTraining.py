
"""
@author: Gavin
"""

'''import statements'''
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

'''Import GAN Model'''
sys.path.append("D:\\Repos\\MiraiGAN")
from GAN_Model2 import *

'''Summarize the performance of the GAN Model by printing the accuracy'''
def summarize_performance(step, g_model, d_model, latent_dim, dataset, n_samples=200):
    X_real = dataset[0]
    Y_real = dataset[1]
    
    preds_fake_old = np.round(d_model.predict(X_fake))
    preds_real_old = np.round(d_model.predict(X_real))
    
    X_fake_new, Y_fake_new = generate_fake_samples(g_model, latent_dim)
    # X_real_new, Y_real_new = get_real_samples(data)
    
    preds_fake_new = np.round(d_model.predict(X_fake_new))
    # preds_real_new = np.round(d_model.predict(X_real_new))
    
    preds_real_train = np.round(d_model.predict(X_train))
                      
    from sklearn.metrics import accuracy_score
    print("Step", step, ": Gen Old Accuracy = {}".format(accuracy_score(Y_fake, preds_fake_old)))
    fake_old_accuracy.append(accuracy_score(Y_fake, preds_fake_old))
    
    print("Step", step, ": Real Test Accuracy = {}".format(accuracy_score(Y_real, preds_real_old)))
    real_new_accuracy.append(accuracy_score(Y_real, preds_real_old))
    
    print("Step", step, ": Gen New Accuracy = {}".format(accuracy_score(Y_fake_new, preds_fake_new)))
    fake_new_accuracy.append(accuracy_score(Y_fake_new, preds_fake_new))
    
    print("Step", step, ": Real Train Accuracy = {}".format(accuracy_score(Y_train, preds_real_train)))
    print()
    



'''Training the generator and discriminator'''
def train(g_model, d_model, gan_model, data_train, data_test, latent_dim, n_epochs=10, n_batch=200):

    # calculate the number of batches per training epoch
    X_train, Y_train = data_train
    bat_per_epo = int(X_train.shape[0] / n_batch)
    
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    #print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (n_epochs, n_batch, half_batch, bat_per_epo, n_steps))
    
    # manually enumerate epochs
    for i in range(n_steps): #n_steps = 1480
        
        # update discriminator (d) 
        
        X_real, y_real = get_real_samples(data_train, n_samples = n_batch) #Y_real is all 1
        y_real = zeros(n_batch)
        # X_real = tf.reshape(X_real, (n_batch, 30, 1))
        d_loss2 = d_model.train_on_batch(X_real, y_real)
        d_loss2_values.append(d_loss2)
        
        
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples = n_batch) #Y_fake is all 0
        y_fake = ones(n_batch)
        # X_fake = tf.reshape(X_fake, (n_batch, 30, 1))
        d_loss1 = d_model.train_on_batch(X_fake, y_fake)
        d_loss1_values.append(d_loss1)
        
        
        # update generator (g)
        X_gan, y_gan = generate_latent_points(latent_dim, half_batch), zeros((half_batch)) #Y_gan is all 1
        # X_gan = tf.reshape(X_gan, (n_batch, 30, 1))
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
        g_loss_values.append(g_loss)
        
        
        # evaluate the model performance every so often
        # if ((i) % (10) == 0):
        summarize_performance(i, g_model, d_model, latent_dim, data_test) 
                
        
        # # summarize loss on this batch
        # print('>%d, d[%.3f,%.3f], g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
        





# Acc/Loss Track
d_loss1_values = []
d_loss2_values = []
g_loss_values = []
fake_old_accuracy = []
fake_new_accuracy = []
real_new_accuracy = []

# Saving Acc/Loss values
np.save('d_loss1_values.npy', d_loss1_values)
np.save('d_loss2_values.npy', d_loss2_values)
np.save('g_loss_values.npy', g_loss_values)
np.save('fake_old_accuracy.npy', fake_old_accuracy)
np.save('fake_new_accuracy.npy', fake_new_accuracy)
np.save('real_new_accuracy.npy', real_new_accuracy)


# size of the latent space
latent_dim = 10 

'''Load the discriminator model from previous training'''
json_file = open('pretrainV1\\pretrain_disc.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
c_model = tf.keras.models.model_from_json(loaded_model_json)
c_model.load_weights("pretrainV1\\pretrain_disc.h5")

'''create the discriminator'''
d_model, _ = build_main_disc()

'''create the generator'''
g_model = build_generator(latent_dim)

'''create the gan'''
gan_model = define_gan(g_model, d_model)


'''load data'''
X_train = np.load('Saved Converted Data (npy)\\data_mirai_train.npy')
Y_train = np.full(len(X_train), 1)
data_train = [X_train, Y_train]
X_test = np.load('Saved Converted Data (npy)\\data_mirai_test.npy')
Y_test = np.full(len(X_test), 1)
data_test = [X_test, Y_test]

'''Pretraining discriminator'''
d_model.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = 10, batch_size = 200, shuffle = True)

'''train model'''
train(g_model, d_model, gan_model, data_train, data_test, latent_dim)



'''Saving Models'''
file_name = "d_model" 
json = file_name + ".json"
h5 = file_name + ".h5"
model_json = d_model.to_json()
with open(json, "w") as json_file:
    json_file.write(model_json)
    d_model.save_weights(h5)
    
file_name = "g_model" 
json = file_name + ".json"
h5 = file_name + ".h5"
model_json = g_model.to_json()
with open(json, "w") as json_file:
    json_file.write(model_json)
    g_model.save_weights(h5)
    
file_name = "gan_model" 
json = file_name + ".json"
h5 = file_name + ".h5"
model_json = gan_model.to_json()
with open(json, "w") as json_file:
    json_file.write(model_json)
    gan_model.save_weights(h5)

'''Opening Models'''
json_file = open('Saved Models\\d_model.json', 'r')
d_model_json = json_file.read()
json_file.close()
d_model = tf.keras.models.model_from_json(d_model_json)
d_model.load_weights("Saved Models\\d_model.h5")

json_file = open('Saved Models\\g_model.json', 'r')
g_model_json = json_file.read()
json_file.close()
g_model = tf.keras.models.model_from_json(g_model_json)
g_model.load_weights("Saved Models\\g_model.h5")

json_file = open('Saved Models\\gan_model.json', 'r')
gan_model_json = json_file.read()
json_file.close()
gan_model = tf.keras.models.model_from_json(gan_model_json)
gan_model.load_weights("Saved Models\\gan_model.h5")
    
    
'''Prepare testing set'''
X_fake, Y_fake = generate_fake_samples(g_model, latent_dim, 200)
X_real, Y_real = get_real_samples(data)
preds_fake = np.round(d_model.predict(X_fake))
preds_real = np.round(d_model.predict(X_real))
np.save('X_fake.npy', X_fake)


'''Metrics'''
from sklearn.metrics import accuracy_score
print("Gen Accuracy = {}".format(accuracy_score(Y_fake, preds_fake)))
print("Real Accuracy = {}".format(accuracy_score(Y_real, preds_real)))




'''EXTRA/TESTING'''
#   1    #
X_real, y_real = get_real_samples(data, 100)
X_real = tf.reshape(X_real, (100, 30, 1))
d_loss1 = d_model.train_on_batch(X_real, y_real)

#   2    #
X_fake, y_fake = generate_fake_samples(g_model, latent_dim, 200) #Y_fake is all 0
d_loss2 = d_model.train_on_batch(X_fake, y_fake)

#   3    #
X_gan, y_gan = generate_latent_points(latent_dim, 200), ones((200, 1)) #Y_gan is all 1
g_loss = gan_model.train_on_batch(X_gan, y_gan)

#   4    #
X_fake_new, Y_fake_new = generate_fake_samples(g_model, latent_dim)
preds_fake_new = np.round(d_model.predict(X_fake_new))
