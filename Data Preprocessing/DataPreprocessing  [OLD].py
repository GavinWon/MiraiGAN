# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 16:57:24 2020

@author: Gavin
"""
# -*- coding: utf-8 -*-
# For Label Encoding


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
dataset.dropna()
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

x = dataset.iloc[:, [2]].values #[id.orig_h] IP
x1 = dataset.iloc[:, [4]] #[id.resp_h] IP
x2 = dataset.iloc[:, [3,5]] #[id.orig_p, id.resp_p]
x3 = dataset.iloc[:, 6:-1].values  #[proto, duration, orig_bytes, orig_pkts, orig_ip_bytes] 
Y = dataset.iloc[:, -1].values


# (unique, counts) = np.unique(x, return_counts=True)
# frequencies = np.asarray((unique, counts)).T

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, Normalizer
le = LabelEncoder()
x[:, 0] = le.fit_transform(x[:,0])
x = x.astype(np.float64)

# One hot Encoding
# ct  = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# x = ct.fit_transform(x).toarray()

#Feature Scaling
sc = StandardScaler()
x = sc.fit_transform(x)
x = sc.fit_transform(x)

threshold = 5
counts = x1["id.resp_h"].value_counts()
repl = counts[counts <= threshold].index
x1 = pd.get_dummies(x1["id.resp_h"].replace(repl, 'uncommon')).values
x1 = (np.argmax(x1, axis=1)).reshape(-1, 1)
x1 = x1.astype(np.float64)
sc1 = StandardScaler()
x1 = sc1.fit_transform(x1)

sc2 = StandardScaler()
x2 = sc2.fit_transform(x2)

le3 = LabelEncoder()
x3[:, 0] = le3.fit_transform(x3[:,0].astype(str))
x3 = x3.astype(np.float64)
sc3 = StandardScaler()
x3 = sc3.fit_transform(x3)

X = np.concatenate((x, x1, x2, x3), axis = 1)
X = X[~np.isnan(X).any(axis=1)]



leY = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y[:-1]
Y = [i + 1 for i in Y]
Y = np.array([float(element) for element in Y])


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)

np.argwhere(np.isnan(x))
np.any(np.isnan(Y))
np.all(np.isfinite(X))



