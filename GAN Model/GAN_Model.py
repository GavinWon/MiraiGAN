
"""
@author: Gavin
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense 
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU, Concatenate
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

'''Construct generator model with given seed_size'''
def build_generator(seed_size):
    
    in_value = Input(shape=seed_size)
    
    hidden1 = Dense(units=20)(in_value)
    batch1 = BatchNormalization()(hidden1) #momentum=0.8
    relu1 = LeakyReLU(alpha=0.2)(batch1)
    
    hidden2 = Dense(units=40)(relu1)
    batch2 = BatchNormalization()(hidden2) #momentum=0.8
    relu2 = LeakyReLU(alpha=0.2)(batch2)
    
    hidden3 = Dense(units=80)(relu2)
    batch3 = BatchNormalization()(hidden3) #momentum=0.8
    relu3 = LeakyReLU(alpha=0.2)(batch3)
    
    out_value = Dense(30)(relu3)
    model = Model(in_value, out_value)
    
    return model
   
    #EXTRA/TEST
    # Output resolution, additional upsampling
    # model.add(Conv2D(128,kernel_size=3,padding="same")) #padding=same
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(LeakyReLU()
    
    # Final CNN layer
    # model.add(Activation("tanh"))

'''Helper MLP for dealing with IP within the Discriminator'''
def build_ip_disc(input_shape=(1214,)): #change input_shape later
    
    in_value = Input(shape=input_shape)
    hidden1 = Dense(units=80, activation='relu')(in_value)
    hidden2 = Dense(units=40, activation='relu')(hidden1)
    hidden3 = Dense(units=30, activation='relu')(hidden2)
    output = Dense(units=20, activation='softmax')(hidden3)
    
    model = Model(inputs=[in_value], outputs=[output])
    # model.compile(loss='linear', optimizer=Adam(lr=0.0001, beta_1=0.5))
    
    return model

'''Main discriminator component'''
def build_main_disc(input_shape=(30,)): #in_shape=(30,1)
    

     #Functional API
    in_value = Input(shape=input_shape)
    
    fe = Dense(32)(in_value) #fe = Conv1D(32, kernel_size=3, strides=2, padding="same")(in_value)
    fe = LeakyReLU(alpha=0.2)(fe)
    
    fe = Dropout(0.25)(fe)
    fe = Dense(64)(fe) #fe = Conv1D(64, kernel_size=3, strides=2, padding="same")(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    
    fe = Dropout(0.25)(fe)
    fe = Dense(128)(fe) #fe = Conv1D(128, kernel_size=3, strides=2, padding="same")(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    
    # fe = Dropout(0.25)(fe)
    # fe = Dense(256)(fe) # fe = Conv1D(256, kernel_size=3, strides=2, padding="same")(fe)
    # fe = BatchNormalization()(fe)
    # fe = LeakyReLU(alpha=0.2)(fe)
    
    # fe = GlobalAveragePooling1D()(fe)
    # fe = Dropout(0.25)(fe)
    # fe = Flatten()(fe)
    
    #determine if real/generated
    d_out_layer = Dense(1, activation='sigmoid')(fe)
    d_model = Model(in_value, d_out_layer)
    d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    
    #determine if benign/malware after determining real
    c_out_layer = Dense(1, activation='sigmoid')(fe)
    c_model = Model(in_value, c_out_layer)
    c_model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate=0.001), metrics=['accuracy']) 
    
    return d_model, c_model
   
    

'''Build a whole discriminator model combining the IP Discriminator and Main Discriminator'''
def build_discriminator(ip_disc, main_disc, inputShape_ip = (1214,), inputShape_other = (10,)):
    
    # main_disc.trainable = False
    
    inTensorIP = Input(inputShape_ip)
    inTensorOther = Input(inputShape_other)
    
    model = ip_disc(inTensorIP)
    
    model = Concatenate()([model, inTensorOther])
    model = main_disc(model)
    # outputFinal = main_disc([outputIP, inTensorOther])
    
    Disc_Model = Model(inputs=[inTensorIP, inTensorOther], outputs=[model])
    opt = Adam(lr=0.0002, beta_1=0.5)
    Disc_Model.compile(loss='binary_crossentropy', optimizer=opt, metrics = ["accuracy"])
    return Disc_Model



    


'''Build the entire GAN connecting Discriminator and Generator'''
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect image output from generator as input to discriminator
	gan_output = d_model(g_model.output)
	# define gan model as taking noise and outputting a classification
	model = Model(g_model.input, gan_output)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model


'''           GETTING AND GENERATING REAL/FAKE SAMPLES                    '''

'''Select a supervised subset of the dataset'''
def get_real_samples(dataset, n_samples=100):
	# split into X and Y
	X, y = dataset
	# choose random instances
	ix = randint(0, X.shape[0], n_samples)
	# select images and labels
	X_sample, y_sample = X[ix], y[ix]
	# generate class labels
	y_other = ones((n_samples, 1)) #1 = REAL DATA
	return X_sample, y_other  

'''Generate points in latent space as input for the generator'''
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	z_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = z_input.reshape(n_samples, latent_dim)
	return z_input

'''Use the generator to generate n fake examples, with class labels'''
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



'''EXTRA/TEST'''
# def desc_ip:
    # model = Sequential()
    # model.add(Dense(32, activation='relu', input_shape = (1224,1)))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dense(20, activation="softmax")) #output layer
    
    
##def build_disc:
     # #determine if real/generated
    # d_model = Sequential()

    # d_model.add(Conv1D(32, kernel_size=3, strides=2, input_shape= (30, 1), padding="same")) #padding=same
    # d_model.add(LeakyReLU(alpha=0.2))

    # d_model.add(Dropout(0.25))
    # d_model.add(Conv1D(64, kernel_size=3, strides=2, padding="same")) #padding=same
    # # model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    # d_model.add(BatchNormalization())
    # d_model.add(LeakyReLU(alpha=0.2))

    # d_model.add(Dropout(0.25))
    # d_model.add(Conv1D(128, kernel_size=3, strides=2, padding="same")) #padding=same
    # d_model.add(BatchNormalization())
    # d_model.add(LeakyReLU(alpha=0.2))

    # d_model.add(Dropout(0.25))
    # d_model.add(Conv1D(256, kernel_size=3, strides=1, padding="same")) #padding=same
    # d_model.add(BatchNormalization())
    # d_model.add(Activation('tanh'))

    # d_model.add(GlobalAveragePooling1D())
    # d_model.add(Dropout(0.25))
    # d_model.add(Flatten())
    
    
    # d_model.add(Dense(1, activation='sigmoid')) #d_out_layer = Lambda(custom_activation)(fe)
    # d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    
    # #determine if benign/malware after determining real

    # c_model = Sequential()

    # c_model.add(Conv1D(32, kernel_size=3, strides=2, input_shape= (30, 1), padding="same")) #padding=same
    # c_model.add(LeakyReLU(alpha=0.2))

    # c_model.add(Dropout(0.25))
    # c_model.add(Conv1D(64, kernel_size=3, strides=2, padding="same")) #padding=same
    # # model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    # c_model.add(BatchNormalization())
    # c_model.add(LeakyReLU(alpha=0.2))

    # c_model.add(Dropout(0.25))
    # c_model.add(Conv1D(128, kernel_size=3, strides=2, padding="same")) #padding=same
    # c_model.add(BatchNormalization())
    # c_model.add(LeakyReLU(alpha=0.2))

    # c_model.add(Dropout(0.25))
    # c_model.add(Conv1D(256, kernel_size=3, strides=1, padding="same")) #padding=same
    # c_model.add(BatchNormalization())
    # c_model.add(Activation('tanh'))

    # c_model.add(GlobalAveragePooling1D())
    # c_model.add(Dropout(0.25))
    # c_model.add(Flatten())
    
    
    # c_model.add(Dense(1, activation='sigmoid'))
    # c_model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate=0.001), metrics=['accuracy']) #default Adam optimizer

