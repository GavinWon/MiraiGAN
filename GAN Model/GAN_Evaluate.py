
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
from tensorflow.keras import backend
from tensorflow.keras.optimizers import Adam

import numpy as np
import sys

import matplotlib.pyplot as plt

'''Import GAN Model'''
sys.path.append("D:\\Repos\\MiraiGAN")
from GAN_Model2 import *

'''Plot Accuracy Values for the Generated/Fake data'''
def plot_fake_accuracy(array, title):
    X = np.arange(len(array)) + 1
    plt.figure(figsize=(20, 16))
    
    plt.plot(X, array, 'bo', markersize = 5)
    
    plt.title(title, fontsize=40, pad=30)
    plt.ylabel('Accuracy', fontsize=30, labelpad=20)
    plt.yticks(np.arange(0.55, 1.05, 0.05), fontsize=20)
    plt.xlabel('Steps', fontsize=30, labelpad=20)
    plt.xticks(np.arange(0, 1600, 100), fontsize=20)
    

    plt.tight_layout()
    plt.savefig('fake.png')
    plt.show()

'''Plot Accuracy Values for the Real data'''
def plot_real_accuracy(array, title):
    X = np.arange(len(array)) + 1
    plt.figure(figsize=(20, 16))
    
    plt.plot(X, array, 'ro', markersize = 5)
    
    plt.title(title, fontsize=40, pad=30)
    plt.ylabel('Accuracy', fontsize=30, labelpad=20)
    plt.yticks(np.arange(0.55, 1.05, 0.05), fontsize=20)
    plt.xlabel('Steps', fontsize=30, labelpad=20)
    plt.xticks(np.arange(0, 1600, 100), fontsize=20)
    

    plt.tight_layout()
    plt.savefig('real.png')
    plt.show()


'''Retrieve the numpy of accuracy values and plot with title'''

fake_new = np.load('Saved Metric Values\\fake_new_accuracy.npy')[2391:] #Ignore first half since values for old model. 
fake_new_title = "Generated/Fake Data Accuracy"
plot_fake_accuracy(fake_new, fake_new_title)

real = np.load('Saved Metric Values\\real_new_accuracy.npy')[2391:] #Ignore first half since values for old model. 
real = real[:600] #Optional: Omit the steps after 600, since accuracy doens't change (straight line)
real_title = "Real Data Accuracy"
plot_real_accuracy(real, real_title)

'''EXTRA'''
# fake_old = np.load('Saved Metric Values\\fake_old_accuracy.npy')[2390:]
# fake_new_title = "Fake Data Accuracy"
# plot_gan_accuracy(fake_old, fake_new_title)