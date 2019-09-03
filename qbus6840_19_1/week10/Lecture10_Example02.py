# -*- coding: utf-8 -*-
"""
Created on Fri May 11 11:23:43 2018

@author: Professor Junbin Gao

Adopted from the program at
http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
"""
import time
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

# We want to be able to repeat the experiment so we fix to a random seed
np.random.seed(70)

# Let us loading data 
data = pd.read_csv('international-airline-passengers.csv', usecols=[1])
data = data.dropna()  # Drop all Nans
data = data.values  # Convert from DataFrame to Python Array  
                    # You need to make sure the data is type of float
                    # you may use data = data.astype('float32') if your data are integers

# Prepare data .....

""" Scaling ...
Neural networks normally work well with scaled data, especially when we use
the sigmoid or tanh activation function. It is a good practice to scale the
data to the range of 0-to-1. This can be easily done by using scikit-learn's 
MinMaxScaler 
"""
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

"""  Splitting ...
We are going to use a time window of k = 3, so we will split the time series as
   x_1, x_2, x_3,     x_4  [prediction]
   x_2, x_3, x_4,     x_5
   x_3, x_4, x_5,     x_6 
   ....
""" 
train_size = int(len(data)*0.67)   # Use the first 2/3 data for training
test_size = len(data) - train_size # the remaining for testing
Dtrain, Dtest = data[0:train_size,:], data[train_size:len(data),:]
# Both Xtrain and Xtest are in time series form, we need split them into sections
# in time-window size 4
time_window = 15
Xtrain, Ytrain = [], []
for i in range(len(Dtrain) - time_window -1):
    Xtrain.append(Dtrain[i:(i+time_window), 0])   # pick up the section in time_window size
    Ytrain.append(Dtrain[i+time_window, 0])       # pick up the next one as the prediction
Xtrain = np.array(Xtrain)    # Convert them from list to array   
Ytrain = np.array(Ytrain) 
 

Xtest, Ytest = [], []
for i in range(len(Dtest) - time_window -1):
    Xtest.append(Dtest[i:(i+time_window), 0])   # pick up the section in time_window size
    Ytest.append(Dtest[i+time_window, 0])       # pick up the next one as the prediction
Xtest = np.array(Xtest)    # Convert them from list to array   
Ytest = np.array(Ytest) 

# We are going to use keras package, so we must reshape our data to the keras required format
# (samples, time_window, features)  we are almost there, but need to reshape into 3D array
# For time series, the feature number is 1 (one scalar value at each time step)
Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], time_window, 1))  
Xtest = np.reshape(Xtest, (Xtest.shape[0], time_window, 1))  

# Define our model .....
model = Sequential()
# Add a LSTM with output_dim (number of hidden neurons) = 100
# input_dim = 1 (for time series) 
#model.add(LSTM(
#        return_sequences=False, input_dim=1,
#        units = 100  #output_dim=100,
#        ))   # Many-to-One model

MyBatchSize = 1

model.add(LSTM(100, input_shape = (time_window,1), batch_size=MyBatchSize,
        return_sequences=False))   # Many-to-One model

# It seems without using batch_size = 1 here causes some problems, a bug?

model.add(Dropout(0.2))            # Impose sparsity as we have so many hidden neurons
# As we will have 100 outputs from LSTM at each time step, we will use a linear 
# layer to map them to a single "prediction" output
model.add(Dense(
        output_dim=1))
model.add(Activation("linear"))

# Compiling model for use
start = time.time()
model.compile(loss="mse", optimizer="rmsprop")
print("Compilation Time : ", time.time() - start)


# Training
model.fit(
	    Xtrain,
	    Ytrain,
	    batch_size=MyBatchSize,
	    nb_epoch=50,   # You increase this number
	    validation_split=0.05)

# Predicting
# make predictions
trainPredict = model.predict(Xtrain,batch_size=MyBatchSize)
testPredict = model.predict(Xtest,batch_size=MyBatchSize)
# invert predictions due to scaling
trainPredict = scaler.inverse_transform(trainPredict)
Ytrain = scaler.inverse_transform(Ytrain[:,np.newaxis])
testPredict = scaler.inverse_transform(testPredict)
Ytest = scaler.inverse_transform(Ytest[:,np.newaxis])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(Ytrain, trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(Ytest, testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# Plotting results
# shift train predictions for plotting
trainPredictPlot = np.empty_like(data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_window:len(trainPredict)+time_window, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(data)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(time_window*2)+1:len(data)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(data), label='True Data')
plt.plot(trainPredictPlot, label='Train Prediction')
plt.plot(testPredictPlot, label='Test Prediction')
plt.legend()
plt.show()

# Overall Prediction with the train model
Xall = [] 
for i in range(len(data) - time_window):
    Xall.append(data[i:(i+time_window), 0])   # pick up the section in time_window size
Xall = np.array(Xall)    # Convert them from list to array  
Xall = np.reshape(Xall, (Xall.shape[0], time_window, 1)) 
allPredict = model.predict(Xall,batch_size=MyBatchSize)
allPredict = scaler.inverse_transform(allPredict)
allPredictPlot = np.empty_like(data)
allPredictPlot[:, :] = np.nan
allPredictPlot[time_window:, :] = allPredict

plt.figure()
plt.plot(scaler.inverse_transform(data), label='True Data')
plt.plot(allPredictPlot, label='One-Step Prediction') 
plt.legend()
plt.show()