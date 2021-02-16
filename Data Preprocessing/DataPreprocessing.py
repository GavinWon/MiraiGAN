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

'''Read dataset using Pandas'''
dataset = pd.read_csv('Data Files\\combined.csv')
dataset.dropna()

'''Details about dataset'''
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



'''One Hot Encoding on x [id.orig_h] (IP Adress), x1 [id.resp_h] (IP Address), 
   Feature Scaling on x2 [id.orig_p, id.orig_p] (Ports)
   Label Encoding on x3 [proto] (Transport Protocol)
   Feature Scaling on x3 [proto, duration, orig_bytes, orig_pkts, orig_ip_bytes] (Integer/Count)'''
x = dataset.iloc[:, [2]].values #Features: [id.orig_h] (IP)
x1 = dataset.iloc[:, [4]] #Features: [id.resp_h] (IP)
x2 = dataset.iloc[:, [3,5]] #Features: [id.orig_p, id.resp_p] (Port)
x3 = dataset.iloc[:, 6:-1].values  #Features: [proto, duration, orig_bytes, orig_pkts, orig_ip_bytes] (Transport Protocol and Integer/Count)
Y = dataset.iloc[:, -1].values #Features: [Benign/Mirai]


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, Normalizer

'''x'''
le = LabelEncoder()
x[:, 0] = le.fit_transform(x[:,0])
x = x.astype(np.float64)
ct  = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = ct.fit_transform(x).toarray()

# sc = StandardScaler()
# x = sc.fit_transform(x)
# x = sc.fit_transform(x)

'''x1'''
threshold = 5 #IPs that appear lower than 5 times will appear in an "uncommon" category
counts = x1["id.resp_h"].value_counts()
repl = counts[counts <= threshold].index
x1 = pd.get_dummies(x1["id.resp_h"].replace(repl, 'uncommon')).values

'''x2'''
sc2 = StandardScaler()
x2 = sc2.fit_transform(x2)

'''x3'''
le3 = LabelEncoder()
x3[:, 0] = le3.fit_transform(x3[:,0].astype(str))
x3 = x3.astype(np.float64)
ct3  = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x3 = ct3.fit_transform(x3)
sc3 = StandardScaler()
x3 = sc3.fit_transform(x3)




'''Combining all the data into one large set'''
X = np.concatenate((x, x1, x2, x3), axis = 1)
X = X[~np.isnan(X).any(axis=1)]

'''Data Check'''
np.argwhere(np.isnan(x))
np.any(np.isnan(Y))
np.all(np.isfinite(X))

'''Split datset into two sub-datasets: containing IP and non-IP type data'''
X_IP = np.concatenate((x, x1), axis = 1)
X_IP = X_IP[~np.isnan(X).any(axis=1)]

X_other = np.concatenate((x2, x3), axis = 1)
X_other = X_other[~np.isnan(X_other).any(axis=1)]

'''Label Encoding on Y [Benign = 0, Mirai = 1]'''
leY = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y[:-1]
# Y = [i for i in Y]
Y = np.array([float(element) for element in Y]) 


'''Splitting into training/testing'''
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)
X_IP_train, X_IP_test = train_test_split(X_IP, test_size = 0.25, random_state = 42)
X_other_train, X_other_test = train_test_split(X_other, test_size = 0.25, random_state = 42)





