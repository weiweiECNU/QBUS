#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 09:21:43 2018

@author: Professor Junbin Gao
"""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from keras.layers.core import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt

np.random.seed(0)

X, y = make_regression(n_samples = 100, n_features = 2, bias = 1.5 )

scaler = MinMaxScaler(feature_range=(0, 1))

X = scaler.fit_transform(X)
y = scaler.fit_transform(y.reshape(-1, 1))

model = Sequential()

n_features = X.shape[1]

model.add(Dense(1, input_dim=n_features, activation='linear', use_bias=True))

model.compile(loss='mean_squared_error', optimizer='adam')

# Alternative if you want to record other metrics in history e.g. accuracy or mean absolute error
# https://keras.io/metrics/
# model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy', 'mae'])

history = model.fit(X, y,  epochs=400, batch_size=16, verbose=2, validation_split=0)

loss_list = history.history['loss']

fig = plt.figure()
plt.plot(loss_list)

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("Advertising.csv", index_col=0)

df.head()

scaler = MinMaxScaler(feature_range=(0, 1))

data = scaler.fit_transform(df.values)

df_scaled = pd.DataFrame(data, columns=df.columns)

df_scaled.head()

y = df_scaled["Sales"]
X = df_scaled[ df_scaled.columns.difference(["Sales"]) ]

X_train, X_test, y_train, y_test = train_test_split(X, y)

n_features = X_train.shape[1]

model_s = Sequential()
model_s.add(Dense(1, input_dim=n_features, activation='linear', use_bias=True))
model_s.compile(loss='mean_squared_error', optimizer='adam')

n_epochs = 400

weights_wrt_is = np.zeros( (n_epochs, 4) )
batch_s = 1

history = None
for i in range(n_epochs):
    if history:
        history = model_s.fit(X_train.values, y_train.values,  epochs= i+1, initial_epoch=i, batch_size=batch_s, verbose=2, validation_split=0)
    else:
        history = model_s.fit(X_train.values, y_train.values,  epochs=1+1, batch_size=batch_s, verbose=2, validation_split=0)
        
    weights = model_s.layers[0].get_weights()
    weights_wrt_is[i, :] =  np.concatenate( (weights[1], weights[0][:,0]) , axis = 0 )

fig = plt.figure()

plt.plot(weights_wrt_is[:, 0])

model_l = Sequential()
model_l.add(Dense(1, input_dim=n_features, activation='linear', use_bias=True))
model_l.compile(loss='mean_squared_error', optimizer='adam')

weights_wrt_il = np.zeros( (n_epochs, 4) )
batch_l = 32

history = None
for i in range(n_epochs):
    if history:
        history = model_l.fit(X_train.values, y_train.values,  epochs= i+1, initial_epoch=i, batch_size=batch_l, verbose=2, validation_split=0)
    else:
        history = model_l.fit(X_train.values, y_train.values,  epochs=1+1, batch_size=batch_l, verbose=2, validation_split=0)
        
    weights = model_l.layers[0].get_weights()
    weights_wrt_il[i, :] =  np.concatenate( (weights[1], weights[0][:,0]) , axis = 0 )
    
fig = plt.figure()

plt.plot(weights_wrt_il[:, 0])

labels = ['beta_0', 'beta_1', 'beta_2', 'beta_3']
colors = ['orange', 'red', 'blue', 'grey']

fig, axarr = plt.subplots(2, 2)

axarr = np.ravel(axarr)

for i in range(4):
    axarr[i].plot(weights_wrt_il[:, i], label = "{0} (16)".format(labels[i]), c = colors[i], linestyle = "solid" )
    axarr[i].plot(weights_wrt_is[:, i], label = "{0} (1)".format(labels[i]), c = colors[i], linestyle = "dashed")
    axarr[i].legend()


# Time Series
from keras.layers.recurrent import LSTM
from keras.layers.core import Activation
from keras.utils.vis_utils import model_to_dot
from IPython.display import Image, display
np.random.seed(0)
# Create the data
n_points = 1000

data_x = np.linspace(1, 100, n_points)
data = np.sin(data_x)
# Plot the data
fig = plt.figure()
plt.plot(data)

time_window = 100

Xall, Yall = [], []

for i in range(time_window, len(data)):
    Xall.append(data[i-time_window:i])
    Yall.append(data[i])
Xall = np.array(Xall)    
       # Convert them from list to array   
Yall = np.array(Yall)

train_size = int(len(Xall) * 0.8)
test_size = len(Xall) - train_size

Xtrain = Xall[:train_size, :]
Ytrain = Yall[:train_size]

Xtest = Xall[-test_size:, :]
Ytest = Yall[-test_size:]

# For time series and LSTM layer we need to reshape into 3D array
Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], time_window, 1))  
Xtest = np.reshape(Xtest, (Xtest.shape[0], time_window, 1))

model = Sequential()

model.add(LSTM(units=50, input_shape=(None, 1), return_sequences=False))
model.add(Dense(1, activation="linear"))

model.compile(loss="mse", optimizer="rmsprop")

# Training
model.fit(Xtrain, Ytrain, batch_size=100, nb_epoch=100, validation_split=0)

dynamic_prediction = np.copy(data[:len(data) - test_size])

for i in range(len(data) - test_size, len(data)):
    last_feature = np.reshape(dynamic_prediction[i-time_window:i], (1,time_window,1))
    next_pred = model.predict(last_feature)
    dynamic_prediction = np.append(dynamic_prediction, next_pred)

dynamic_prediction = dynamic_prediction.reshape(-1,1)

fig = plt.figure()
plt.plot(data, label = "Original")
plt.plot(dynamic_prediction, label = "Predicted")
plt.legend(loc="lower left")


