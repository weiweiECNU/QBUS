# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 12:21:22 2018

@author: Professor Junbin Gao

"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import statsmodels as sm 
import statsmodels.api as smt
from datetime import datetime

# Make sure we can replicate the results
np.random.seed(12345)

# It seems I cannot find out an ARIMA process function similar to ArmaProcess
# Since statsmodels version 0.8, an ARIMA model can be produced with seasonal ARIMA process
# in terms of state space process  

# This example shows you how to define an ARIMA process by using SARIMAX. You must update
# statsmodels to version 0.8 to run this program

# Prepare the data which is the American Wholesale price index (WPI) 
data = pd.read_csv('Wholesale_price_index.csv')
# We need use time index for the data
data.index = pd.to_datetime(data.t)
# You may use Variable explorer to see what are in data 

# First let us analyse the ACF and PACF of data
smt.graphics.tsa.plot_acf(data['wpi'], lags=30, alpha = 0.05)
smt.graphics.tsa.plot_pacf(data['wpi'], lags=30, alpha = 0.05)
# ACF does die down quickly, so it is not stationary

# We try the first order difference first
data['D.wpi'] = data['wpi'].diff()
# Graph data
fig, axes = plt.subplots(1, 2, figsize=(15,4))

# Levels
axes[0].plot(data.index._mpl_repr(), data['wpi'], '-')
axes[0].set(title='US Wholesale Price Index')

# Log difference
axes[1].plot(data.index._mpl_repr(), data['D.wpi'], '-')
axes[1].hlines(0, data.index[0], data.index[-1], 'r')
axes[1].set(title='US Wholesale Price Index - difference');
    

# Check stationarity for differenced data
smt.graphics.tsa.plot_acf(data['D.wpi'][1:], lags=30 )
smt.graphics.tsa.plot_pacf(data['D.wpi'][1:], lags=30)
# It seems ACF dies down.  The lag of AR = 1 and MA = 4


# Let us try the ARIMA(1,1,1) model for this data
# Or equivalently the postulated data process is then:
# Δyt=c+ϕ1Δyt−1+θ1ϵt−1+ϵt
# Note we need use data['wpi'] not data['D.wpi'] because d = 1
model1 = smt.tsa.statespace.SARIMAX(data['wpi'], trend='c', order=(1,1,4)) 

# Now let us fit the data to the defined model ARIMA(1,1,1)
res1 = model1.fit(disp=False)
print(res1.summary())
# The fitted model is
#Δyt=0.1050+0.8740Δy{t−1}−0.4206ϵ{t−1}+ϵt 
 
# We can also try the log transformed data and differencing
data['D.ln_wpi'] = data['ln_wpi'].diff()
# Graph data
fig, axes = plt.subplots(1, 2, figsize=(15,4))

# Levels
axes[0].plot(data.index._mpl_repr(), data['ln_wpi'], '-')
axes[0].set(title='US Wholesale Price Index (logs)')

# Log difference
axes[1].plot(data.index._mpl_repr(), data['D.ln_wpi'], '-')
axes[1].hlines(0, data.index[0], data.index[-1], 'r')
axes[1].set(title='US Wholesale Price Index - difference of logs');
    

# Check acf and pacf. In this case we draw them together
fig, axes = plt.subplots(1, 2, figsize=(15,4))

fig = smt.graphics.tsa.plot_acf(data.ix[1:, 'D.ln_wpi'], lags=100, ax=axes[0])
fig = smt.graphics.tsa.plot_pacf(data.ix[1:, 'D.ln_wpi'], lags=40, ax=axes[1])

# Once again it could be a ARIMA(1, 1, 4).
model2 = smt.tsa.statespace.SARIMAX(data['ln_wpi'], trend='c', order=(1,1,4))
res2 = model2.fit(disp=False)
print(res2.summary())

# Please check out the estimate coeffficients.

# Finally we can make use of fitted model to forecast, e.g., forecast to 2000-01-01
predict = res2.get_prediction(end='2000-01-01')
# The forecasts are in
fc = predict.predicted_mean
plt.figure()
plt.plot(fc[10:])