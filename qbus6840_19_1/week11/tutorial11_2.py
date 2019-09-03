#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:08:06 2017

@author: steve
"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential

# We want to be able to repeat the experiment so we fix to a random seed
np.random.seed(1)

# Let us loading data 
data = pd.read_csv('AirPassengers.csv', usecols=[1])
data = data.dropna()

scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

time_window = 20

#%%
Xall, Yall = [], []

for i in range(time_window, len(data)):
    Xall.append(data[i-time_window:i, 0])
    Yall.append(data[i, 0])
Xall = np.array(Xall)    # Convert them from list to array   
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

#%%
# Define our model .....
model = Sequential()
# Add a LSTM with units (number of hidden neurons) = 50
# input_dim = 1 (for time series)
# return sequences = False means only forward the last lagged output to the
# following layer

#model.add(LSTM(unit_number,
#           batch_input_shape=(batch_size, n_prev, 1),
#           forget_bias_init='one',
#           return_sequences=True,
#           stateful=True))

model.add(LSTM(
        input_shape=(None, 1),
        units=50,
        return_sequences=False))   # Many-to-One model
model.add(Dense(
        output_dim=1))
model.add(Activation("linear"))

model.compile(loss="mse", optimizer="rmsprop")
#%%
# Training
from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
model.fit(
	    Xtrain,
	    Ytrain,
	    batch_size=5,
	    nb_epoch=0,
	    validation_split=0.1)

allPredict = model.predict(np.reshape(Xall, (124,20,1)))
allPredict = scaler.inverse_transform(allPredict)
allPredictPlot = np.empty_like(data)
allPredictPlot[:, :] = np.nan
allPredictPlot[time_window:, :] = allPredict

plt.figure()
plt.plot(scaler.inverse_transform(data), label='True Data')
plt.plot(allPredictPlot, label='One-Step Prediction') 
plt.legend()
plt.show()

trainScore = math.sqrt(mean_squared_error(Ytrain, allPredict[:train_size,0]))
print('Training Data RMSE: {0:.2f}'.format(trainScore))
#%%
dynamic_prediction = np.copy(data[:len(data) - test_size])

for i in range(len(data) - test_size, len(data)):
    last_feature = np.reshape(dynamic_prediction[i-time_window:i], (1,time_window,1))
    next_pred = model.predict(last_feature)
    dynamic_prediction = np.append(dynamic_prediction, next_pred)

dynamic_prediction = dynamic_prediction.reshape(-1,1)
dynamic_prediction = scaler.inverse_transform(dynamic_prediction)
    
plt.figure()
plt.plot(scaler.inverse_transform(data[:len(data) - test_size]), label='Training Data')
plt.plot(np.arange(len(data) - test_size, len(data), 1), scaler.inverse_transform(data[-test_size:]), label='Testing Data')
plt.plot(np.arange(len(data) - test_size, len(data), 1), dynamic_prediction[-test_size:], label='Out of Sample Prediction') 
plt.legend(loc = "upper left")
plt.show()

testScore = math.sqrt(mean_squared_error(Ytest, dynamic_prediction[-test_size:]))
print('Dynamic Forecast RMSE: {0:.2f}'.format(testScore))