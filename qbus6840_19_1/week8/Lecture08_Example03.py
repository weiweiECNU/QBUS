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

# This example shows you how to define an ARIMA process by using legacy module ARIMA. 
# We cannot use the given coefficients to define an ARIMA, instead we specify the lags for
# a given time series and estimate it and then simulate it.
 
# First we load the data
data = pd.read_csv('Example08_03.csv',header=None)
# The data was simulated from
# (y_t - y_{t-1}) =  -0.1 (y_{t-1} - y_{t-2}) + 0.2(y_{t-2} - y_{t-3}) + 0.1 (y_{t-3} - y_{t-4}) - 0.7(y_{t-4} - y_{t-5})
#  + e_t + 0.4 e_{t-1} - 0.4e_{t-2} + 0.3 e_{t-3} + 0.3 e_{t-4} + 0.1 e_{t-5} 
# That is ARIMA(4,1,5)
data.index = pd.to_datetime(data.index)
 
arima_model = sm.tsa.arima_model.ARIMA(data[0].values, order=(4,1,5))        
result = arima_model.fit(disp=False)
print(result.summary())

# This is the first order differenced data
z = arima_model.predict(result.params,start=1001, end=1050)   # 1, 1000
# The predicted data
y = np.cumsum(z) + data.iloc[999,0]
 
plt.figure()
plt.plot(data[0].values)

smt.graphics.tsa.plot_acf(data[0].values, lags=30, alpha = 0.05)
# Because this is an MA model of lag 2, the ACF dies out after lag 2
smt.graphics.tsa.plot_pacf(data[0].values, lags=30, alpha = 0.05)

plt.figure()
plt.plot(y)
smt.graphics.tsa.plot_acf(y, lags=30, alpha = 0.05)
# Because this is an MA model of lag 2, the ACF dies out after lag 2
smt.graphics.tsa.plot_pacf(y, lags=30, alpha = 0.05) 