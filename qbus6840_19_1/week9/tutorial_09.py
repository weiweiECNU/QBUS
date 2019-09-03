#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 16:28:03 2017

@author: steve
@Update: Boyan
"""

import matplotlib.pyplot as plt
import statsmodels as sm 
import statsmodels.api as smt
import numpy as np
import pandas as pd


"""
StatsModel variations
"""

# http://nipy.bic.berkeley.edu/nightly/statsmodels/doc/html/tsa.html

# We have two options when it comes to using statsmodel:
# 1. Directly generate synthetic samples from an ARIMA model object
# 2. Create an ARIMA process object, which we can analyse and generate samples from

# In Tutorial 08 we used #1
# However we would like to do some analysis of our models so in general it's
# probably a better idea to use ARIMA process objects instead of ARIMA models


# Setting up the random generator 
np.random.seed(12345)

arparams = np.array([0.9])
zero_lag = np.array([1])
maparams = np.array([0.6, -0.5])

ar = np.r_[1, -arparams]  # add zero-lag (coefficient 1) and negate
ma = np.r_[1, maparams]   # add zero-lag

#%%
c = 0

sigma2 = 1

"""
MA Model
"""

ma_model = sm.tsa.arima_process.ArmaProcess(ar = zero_lag, ma = ma)

# Check if it is stationary
print("MA Model is{0}stationary".format(" " if ma_model.isstationary else " not " ))

# Check if it is invertible
print("MA Model is{0}invertible".format(" " if ma_model.isinvertible else " not " ))

# Plot ACF
plt.figure()
plt.stem(ma_model.acf())

# Plot PACF
plt.figure()
plt.stem(ma_model.pacf())

#%%
# Generate samples
n_samples = 250

#Yt = c + e_t + 0.6e_{t-1} - 0.5 e_{t-2}
y2 = c + ma_model.generate_sample(n_samples)

plt.figure()
plt.plot(y2)
plt.title("Moving Average (MA) Model")

# Notice that we can produce the ACF and PACF *BEFORE* we generated any samples
# Let's compare with the sample ACF and PACF

smt.graphics.tsa.plot_acf(y2, lags=30, alpha = 0.05)
smt.graphics.tsa.plot_pacf(y2, lags=30, alpha = 0.05)

# Mean
# Since epsilon_t = y_t - y_t-1 and y_t - y_t-1
# then E(epsilon_t) = 0

y2_mean = c

# Variance
# c is constant so var(c) = 0
# var(epsilon_t) = sigma^2
# then var(y2) = 0 + sigma^2 + theta_1 * sigma^2 + theta_2 * sigma^2

y2_variance = sigma2 * (1 + maparams[0]**2 + maparams[1]**2)

#%%
"""
ARMA Model
"""

arma_model = sm.tsa.arima_process.ArmaProcess(ar = ar, ma = ma)

n_samples = 1000

#Yt = c + 0.9Y_{t-1} + 0.6e_{t-1} - 0.5 e_{t-2} + e_t 
y3 = 10 + arma_model.generate_sample(n_samples)

plt.figure()
plt.plot(y3)
plt.title("ARMA Model")

smt.graphics.tsa.plot_acf(y3, lags=30, alpha = 0.05)
smt.graphics.tsa.plot_pacf(y3, lags=30, alpha = 0.05)

#%%
"""
Model Fitting
"""

# Lets try and fit the model back
# (0, 2) refers to the number of parameters in the (AR, MA) parts selectively

model_y2 = sm.tsa.arima_model.ARMA(y2, (0, 2)).fit(trend='c')
print("Estimated MA Model Parameters: " + str(model_y2.params))

model_y3 = sm.tsa.arima_model.ARMA(y3, (1, 2)).fit(trend='c')
print("Estimated ARMA Model Parameters: " + str(model_y3.params))

forecast = model_y3.predict(start=1, end=1500)
plt.figure()
plt.plot(forecast)

#%% load the dataset
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

smt.graphics.tsa.plot_acf(ts_log_diff, lags=30, alpha = 0.05)
smt.graphics.tsa.plot_pacf(ts_log_diff, lags=30, alpha = 0.05)

#%% AR(2) model
from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(ts_log, order=(2, 1, 0))  
results_AR = model.fit(disp=-1) 
residuals_AR = pd.DataFrame(results_AR.resid) 
plt.figure() 
plt.plot(residuals_AR)
plt.title('AR(2) RSS: %.4f'% sum((results_AR.resid.values)**2))

# Get Fitted Series
fitted = results_AR.predict(typ = 'levels', dynamic = False)

# Actual vs Fitted 
results_AR.plot_predict(dynamic=False) 
plt.show()

#%% Example of  MA(1) model
model = ARIMA(ts_log, order=(0, 1, 1))  
results_MA = model.fit(disp=-1)  

residuals_MA = pd.DataFrame(results_MA.resid) 
plt.figure() 
plt.plot(residuals_MA)
plt.title('MA(1) RSS: %.4f'% sum((results_MA.resid.values)**2))

# Get Fitted Series
fitted = results_MA.predict(typ = 'levels', dynamic = False)

# Actual vs Fitted 
results_MA.plot_predict(dynamic=False) 
plt.show()
plt.title("ARIMA(2,1,0) model fitted results")

#%% Example of ARIMA(2,1,1)
model = ARIMA(ts_log, order=(2, 1, 1))  
results_ARIMA = model.fit(disp=-1)  

residuals = pd.DataFrame(results_ARIMA.resid) 
plt.figure() 
plt.plot(residuals)
plt.title('ARIMA(2,1,1) RSS: %.4f'% sum((results_ARIMA.resid.values)**2))

# Get Fitted Series
fitted = results_ARIMA.predict(typ = 'levels', dynamic = False)

# Actual vs Fitted 
results_ARIMA.plot_predict(dynamic=False) 
plt.show()

#%%
plt.figure()
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('ARIMA(2,1,1) RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))

##%% Below are the example on how to scale back to the original data.
#predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
#print(predictions_ARIMA_diff.head())
#
#predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
#print(predictions_ARIMA_diff_cumsum.head())
#
#predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
#predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
#
#predictions_ARIMA = np.exp(predictions_ARIMA_log)
#plt.figure()
#plt.plot(ts,label="Original data")
#plt.plot(predictions_ARIMA,label = "ARIMA fitted")
#plt.title('ARIMA(2,1,1) RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
#plt.legend()

#%% Reselect the order for ARIMA
import statsmodels.tsa.stattools as st
order = st.arma_order_select_ic(ts_log_diff,max_ar=15,max_ma=15,ic=['aic'])
print(order.aic_min_order)

#%%
#p,q = order.aic_min_order[0], order.aic_min_order[1]
# or you can directly assign p,q with 15,0 without training
p,q = 15,0

model_AIC = ARIMA(ts_log, order=(p, 1, q))  
results_AIC_ARIMA = model_AIC.fit(disp=-1)  

#%%
residuals_AIC = pd.DataFrame(results_AIC_ARIMA.resid) 
plt.figure() 
plt.plot(residuals_AIC)
plt.title('ARIMA(15,1,0) RSS: %.4f'% sum((results_AIC_ARIMA.resid.values)**2))

# Get Fitted Series
fitted_AIC = results_AIC_ARIMA.predict(typ = 'levels', dynamic = False)

# Actual vs Fitted 
results_AIC_ARIMA.plot_predict(dynamic=False) 
plt.show()
plt.title("ARIMA(15,1,0) model fitted results")