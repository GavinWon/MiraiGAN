# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 23:30:34 2021

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

import matplotlib.pyplot as plt
from tensorflow.keras import backend

sys.path.append("D:\\Repos\\MiraiGAN")
from GAN_Model2 import *

