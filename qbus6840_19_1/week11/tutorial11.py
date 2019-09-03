#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 12:31:47 2017

@author: steve
"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.layers.core import Dense 
from keras.models import Sequential

np.random.seed(1)

# Let us loading data 
data = pd.read_csv('AirPassengers.csv', usecols=[1])
data = data.dropna()  # Drop all Nans
data = data.values  # Convert from DataFrame to Python Array  
                    # You need to make sure the data is type of float
                    # you may use data = data.astype('float32') if your data are integers

plt.figure()
plt.plot(data)

#%%
# Prepare data .....

""" Scaling ...
Neural networks normally work well with scaled data, especially when we use
the sigmoid or tanh activation function. It is a good practice to scale the
data to the range of 0-to-1. This can be easily done by using scikit-learn's 
MinMaxScaler 
"""
scaler = MinMaxScaler(feature_range=(0, 1))

data = scaler.fit_transform(data)

#%%

"""  Splitting ...
We are going to use a time lag p = 3, so we will split the time series as
    [FEATURES]             TARGET (PREDICTION)
   [x_1, x_2, ..., x_12],    x_13
   [x_2, x_3, ..., x_13],    x_14
   [x_3, x_4, ..., x_14],    x_15 
   ....
""" 

time_window = 12

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

#%%
# Define our model 
model = Sequential()

model.add(Dense(20, input_dim=time_window, activation='relu'))  # sigmoid

#model.add(Dense(20, activation='sigmoid'))  # relu

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')   #adam

#%% If you want to plot the model
# IF you have problem with visualize the model, please refer to the following tutorial:
# https://github.com/keras-team/keras/issues/12538 
# step1: unistall pydot "conda uninstall pydot"
# step2: install pydot-ng: "pip install pydot-ng"
# step3: change the code in '...\keras\utils\vis_utils.py' 
#        from "import pydot" to "import pydot_ng as pydot"
from keras.utils import plot_model
plot_model(model, to_file='model_NN.png', show_shapes = True, show_layer_names = True)

#%%
from keras.utils.vis_utils import model_to_dot
from sklearn.externals.six import StringIO

import pydotplus

from IPython.display import Image, display

f = StringIO()

dot_obj = model_to_dot(model)

#
display(Image(dot_obj.create_png()))

#%%

# Training
model.fit(Xtrain, Ytrain,  nb_epoch=100, batch_size=2, verbose=2, validation_split=0)

# One-step, in sample prediction

allPredict = model.predict(Xall)
allPredictPlot = scaler.inverse_transform(allPredict)

plt.figure()
plt.plot(scaler.inverse_transform(data), label='True Data')
plt.plot(np.arange(time_window, len(data)),allPredictPlot, label='One-Step Prediction') 
plt.legend()

trainScore = math.sqrt(mean_squared_error(Ytrain, allPredict[:train_size,0]))
print('Training Data RMSE: {0:.2f}'.format(trainScore))
#%%
# Dynamic, out of sample prediction
# We need to compute each timeseries point value then feed it
# back into the prediction function as part of the next 12 time lags 

dynamic_prediction = np.copy(data[:len(data) - test_size])

for i in range(len(data) - test_size, len(data)):
    last_feature = np.reshape(dynamic_prediction[i-time_window:i], (1,time_window))
    next_pred = model.predict(last_feature)
    dynamic_prediction = np.append(dynamic_prediction, next_pred)

dynamic_prediction = dynamic_prediction.reshape(-1,1)
dynamic_prediction = scaler.inverse_transform(dynamic_prediction)
    
plt.figure()
plt.plot(scaler.inverse_transform(data[:len(data) - test_size]), label='Training Data')
plt.plot(np.arange(len(data) - test_size, len(data), 1), scaler.inverse_transform(data[-test_size:]), label='Testing Data')
plt.plot(np.arange(len(data) - test_size, len(data), 1), dynamic_prediction[-test_size:], label='Out of Sample Prediction') 
plt.legend(loc = "upper left")

testScore = math.sqrt(mean_squared_error(Ytest, dynamic_prediction[-test_size:]))
print('Dynamic Forecast RMSE: {0:.2f}'.format(testScore))


