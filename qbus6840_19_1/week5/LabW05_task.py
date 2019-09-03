#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:35:32 2019

@author: Boyan Zhang
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

data = pd.read_csv('AirPassengers.csv')

from datetime import datetime
con=data['Month']
data['Month']=pd.to_datetime(data['Month'])

data.set_index('Month', inplace=True)
#check datatype of index
print(data.index)

ts = data['Passengers']
ts.head(10)

#%% calculate the trend + cycle
Trend = ts.rolling(12, center = True).mean().rolling(2,center = True).mean()
Trend = Trend.shift(-1)
#%% calculate the seasonal components
Seasonal = (ts + 0.)/ Trend

plt.figure()
plt.plot(ts)
plt.plot(Trend)
plt.figure()
plt.plot(Seasonal)

#%%
Seasonal = np.nan_to_num(Seasonal)
monthly_S = np.reshape(Seasonal, (12,12))
Seasonal_index = np.mean(monthly_S[1:11,:], axis=0)

#%%
c = 12 / sum(Seasonal_index)
Seasonal_index = Seasonal_index*c

#%% seasonal adjusted data
tiled_avg = np.tile(Seasonal_index, 12)
seasonally_adjusted = (ts + 0.) /tiled_avg
plt.figure()
plt.plot(seasonally_adjusted)

#%% reestimate trend by using linear regression
X = np.linspace(1, len(ts), len(ts))
Y = ts
X = np.reshape(X, (len(X), 1))
Y = np.reshape(Y, (len(Y), 1))

X_train = X[:-12]
X_test = X[-12:]
Y_train = ts[:-12]
Y_test = ts[-12:]
#%% Train a linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train, Y_train)
trend_test = lr.predict(X_test)
trend_train = lr.predict(X_train)
plt.figure()
plt.plot(X,Y)
plt.plot(X_test,trend_test)
plt.plot(X_train,trend_train)
#%%
trend_new = lr.predict(X)
residual= (0. + ts)/trend_new/tiled_avg

# check the stationary (advanced)
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

test_stationarity(residual)
#%%
forecast = trend_test*Seasonal_index*1.
plt.figure()
plt.plot(X,ts)
plt.plot(X_test,forecast)
