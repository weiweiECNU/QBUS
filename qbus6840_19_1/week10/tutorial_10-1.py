# -*- coding: utf-8 -*-
"""
Created on May  2 23:12:24 2019

@author: Boyan Zhang
"""
#%% Task 1
import matplotlib.pyplot as plt
import statsmodels as sm 

import statsmodels.api as smt
import numpy as np
import pandas as pd

maparams = np.array([0.7, 0,0,0,0,0,0,0,0,0,0, 0.8, 0.8*0.7])
ma = np.r_[1, maparams]   # add zero-lag
zero_lag = np.array([1])

#%%

ma_model = sm.tsa.arima_process.ArmaProcess(ar = zero_lag, ma = ma)

# Plot ACF
plt.figure()
plt.stem(ma_model.acf()[:15])
plt.title("ACF curve for SARIMA(0,0,1)(0,0,1)12")
plt.show()

#%% Task 2
arparams = np.array([0.7, 0,0,0,0,0,0,0,0,0,0, 0.8, -0.8*0.7])
ar = np.r_[1, -arparams]   # add zero-lag
ar_model = sm.tsa.arima_process.ArmaProcess(ar = ar, ma = zero_lag)

plt.figure()
plt.stem(ar_model.pacf()[:30])
plt.title("PACF curve for ARIMA(1,0,0)(1,0,0)12")
plt.show()

#%% Task 3
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
airdata = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month',date_parser=dateparse)
ts = airdata['Passengers']
plt.figure()
plt.plot(ts)
plt.title("Air passenger data")

#%% log the data
ts_log = np.log(ts)
plt.figure()
plt.plot(ts_log)
plt.title("Air passenger data (log)")

#%% take the 1st order diff
ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)
plt.figure()
plt.plot(ts_log_diff)
plt.title("Air passenger data (log-1st diff)")

#%%
import statsmodels as sm 
import statsmodels.api as smt
from statsmodels.tsa.statespace.sarimax import SARIMAX

smt.graphics.tsa.plot_acf(ts_log_diff, lags=30, alpha = 0.05)
smt.graphics.tsa.plot_pacf(ts_log_diff, lags=30, alpha = 0.05)

#%%
train_ratio = 0.7
split_point = round(len(ts_log)*train_ratio)
training, testing = ts_log[0:split_point], ts_log[split_point:]

#%%
model = SARIMAX(training, \
               order=(2,1,4),\
               seasonal_order=(1,0,0,12),\
               enforce_stationarity=False,\
               enforce_invertibility=False)

model_fit = model.fit(disp=-1) 

forecast = model_fit.forecast(len(testing))
plt.figure()
plt.plot(np.exp(forecast),'r')
plt.plot(np.exp(ts_log),'b')
plt.title('SARIMA(2,1,4)(1,0,0) RSS: %.4f'% sum((model_fit.resid.values)**2))
plt.xlabel("Years")
plt.ylabel("Passengers")
plt.axvline(x=ts_log.index[split_point],color='black')
