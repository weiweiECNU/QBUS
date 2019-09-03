
'''
Libs
'''
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

import statsmodels as sm 
import statsmodels.api as smt

import warnings
warnings.simplefilter('ignore')

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 16, 4 # set fig width and height in inches





np.random.seed(1) 

# Read the data
data = pd.read_csv('international-airline-passengers.csv', usecols=[1]) 
data = data.dropna()
data = data.values 

plt.figure()
plt.plot(data)
plt.show()

scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# Prepare training and validation set
time_window = 12 # the determined lag "p"

Xall, Yall = [], []
for i in range(time_window, len(data)): 
    Xall.append(data[i-time_window:i, 0]) 
    Yall.append(data[i, 0]) 
    
    
# Convert data type from list to array
Xall = np.array(Xall)    
Yall = np.array(Yall)

# Train-test split
train_size = int(len(Xall) * 0.8)
test_size = len(Xall) - train_size
Xtrain = Xall[:train_size, :]
Ytrain = Yall[:train_size]
Xtest = Xall[-test_size:, :]
Ytest = Yall[-test_size:]


trainScore = []
testScore = []

model = Sequential()

for c in range(15,26):

    model.add(Dense(c, input_dim=time_window, activation='relu')) # specify width (the number of neurons) to be 20
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(Xtrain, Ytrain, epochs=100, batch_size=32, verbose=0, validation_split=0.05)
    allPredict = model.predict(Xall) 
    allPredictPlot = scaler.inverse_transform(allPredict) 


    trainScore.append(math.sqrt(mean_squared_error(Ytrain, allPredict[:train_size,0]))) 


    dynamic_prediction = np.copy(data[:len(data) - test_size])

    for i in range(len(data) - test_size, len(data)):
        last_feature = np.reshape(dynamic_prediction[i-time_window:i], (1,time_window)) 
        next_pred = model.predict(last_feature) 
        dynamic_prediction = np.append(dynamic_prediction,next_pred) 

    dynamic_prediction = dynamic_prediction.reshape(-1,1) 
    dynamic_prediction = scaler.inverse_transform(dynamic_prediction)


    testScore.append(math.sqrt(mean_squared_error(Ytest, dynamic_prediction[-test_size:])))





plt.figure()
plt.plot(range(15,26), trainScore, label='Training RMSE')
plt.plot(range(15,26), testScore, label='Test RMSE')
plt.legend(loc = "upper left")




