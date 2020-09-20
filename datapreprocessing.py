# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 21:20:11 2020

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


dataset = pd.read_csv('New\\combined.csv')
dataset.nunique()
dataset.dtypes
# ts                68262
# uid              106394
# id.orig_h           308
# id.orig_p         29996
# id.resp_h          3666
# id.resp_p           459
# proto                 3
# duration          64750
# orig_bytes         6678
# orig_pkts           636
# orig_ip_bytes      6661
# Label                 2

print(dataset.head())

x = dataset.iloc[:, [2,3,5]].values #[id.orig_h, id.orig_p, id.resp_p]
x1 = dataset.iloc[:, [4]] #[id.resp_h]
x2 = dataset.iloc[:, 6:-1].values
Y = dataset.iloc[:, -1].values

np.any(np.isnan(x2))
np.all(np.isfinite(x2))

# (unique, counts) = np.unique(x, return_counts=True)
# frequencies = np.asarray((unique, counts)).T

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
le = LabelEncoder()
x[:, 0] = le.fit_transform(x[:,0])
x = x.astype(np.float64)
ct  = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = ct.fit_transform(x).toarray()

threshold = 5
counts = x1["id.resp_h"].value_counts()
repl = counts[counts <= threshold].index
x1 = pd.get_dummies(x1["id.resp_h"].replace(repl, 'uncommon')).values

le = LabelEncoder()
x2[:, 0] = le.fit_transform(x2[:,0].astype(str))
x2 = x2.astype(np.float64)
ct2  = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x2 = ct2.fit_transform(x2)

X = np.concatenate((x, x1, x2), axis = 1)



from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)




def CountFrequency(my_list):
    freq = {}
    for item in my_list: 
        if (item in freq): 
            freq[item] += 1
        else: 
            freq[item] = 1
    return freq