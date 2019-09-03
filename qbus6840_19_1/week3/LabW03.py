#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:35:32 2019

@author: boyanzhang
"""
# load the data

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

data = pd.read_csv('AirPassengers.csv')

print(data.head())
print('\n Data Types:')
print(data.dtypes)

#%%
# 
from datetime import datetime
con=data['Month']
data['Month']=pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)
#check datatype of index
print(data.index)

ts = data['Passengers']
ts.head(10)

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
plt.figure()
plt.plot(ts)

#%%
# Rolling windows
rolling_data = ts.rolling(12,center=True)
plt.figure()
plt.plot(ts,'r-',label="time series data")
plt.plot(rolling_data.mean(),'b-',label="Rolling mean")
plt.plot(rolling_data.std(),'g-',label="Rolling std")
plt.legend()

#%%
# make the series stationary
ts_log= np.log(ts)
plt.figure()
plt.plot(ts_log, color='red',label='log')

#%%
Trend = ts_log.rolling(2, center = True).mean().rolling(12,center = True).mean()
np.random.seed(0)

plt.figure()
plt.plot(ts_log, color='red',label='log')
plt.plot(Trend, color='blue',label='MA')
plt.title('Initial TREND estimate ')
plt.xlabel('Month')
plt.ylabel('Number')

#%%

ts_res = ts_log - Trend
ts_res.dropna(inplace = True)
ts_res = np.nan_to_num(ts_res)

plt.figure()
plt.plot(ts_res)
plt.title('Seasonal')
plt.xlabel('Month')

#%%
# Rolling windows
rolling_data_res = pd.Series(ts_res, dtype='float64').rolling(12,center=True)
plt.figure()
plt.plot(ts_res,'r-',label="time series data")
plt.plot(rolling_data_res.mean(),'b-',label="Rolling mean")
plt.plot(rolling_data_res.std(),'g-',label="Rolling std")
plt.legend()

#%%
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

def plot_curve(timeseries):
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)
    #Plot rolling statistics:
    plt.figure()
    plt.plot(timeseries, color='blue',label='time series data')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
#%%
test_stationarity(ts)
plot_curve(ts)

#%%
test_stationarity(ts_res)
plot_curve(ts_res)
