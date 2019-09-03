#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:28:55 2017

@author: steve
Updated by: Boyan
"""
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import statsmodels as sm 
import statsmodels.api as smt
# We will use statsmodel's time series plots to draw ACF etc.
from pandas.plotting import autocorrelation_plot
# Or we can use pandas to draw ACF

# This data is of type DataFrame with only one column
data = pd.read_csv('data.csv')
# We make it as a Series by taking out the column
data = data['Data']

# Plot our data
plt.figure()
plt.plot(data)

#%%
# We calculate difference series, data[1]-data[0], data[2]-data[1],...
diff_data = pd.Series.diff(data)
# Checking the first entry in diff_data
diff_data.iloc[0]
# Why is it a nan?
diff_data = diff_data.dropna()
# If we dont do this, what will happen for the following codes

# Plot the differenced data
plt.figure()
plt.plot(diff_data)

# Plot the ACF for the data. This call opens a new plot
smt.graphics.tsa.plot_acf(data, lags=30, alpha = 0.02)

# lags = 30 means drawing 30 lags
# Here alpha=.05, 95% confidence intervals are returned 
# where the standard deviation is computed according to 
# Bartlett’s formula.
# You may change 0.05 to other values for alpha to see what will happen

# Plot the PACF for the data. This call opens a new plot
smt.graphics.tsa.plot_pacf(data, lags=30, alpha=0.02)
plt.show()
#%%
# For differenced time series
smt.graphics.tsa.plot_acf(diff_data, lags=30, alpha = 0.05)
smt.graphics.tsa.plot_pacf(diff_data, lags=30, alpha = 0.05)
plt.show()
#%%
# Use pandas for ACF, but not plot function for PACF
plt.figure()  # We need prepare the figure
autocorrelation_plot(data)

#%%
"""
AR Model
"""

# Setting up the random generator 
np.random.seed(1)

arparams = np.array([0.9])
zero_lag = np.array([1])

ar = np.r_[1, -arparams]  # add zero-lag (coefficient 1) and negate

c = 0

sigma2 = 1
   
# Yt = c + 0.9Y_{t-1} + e_t          
y = c + sm.tsa.arima_process.arma_generate_sample(ar = ar, ma = zero_lag, nsample = 10000)

plt.figure()
plt.plot(y)
plt.title("Autoregressive (AR) Model")

#%%
# Examine the ACF and PACF plots

smt.graphics.tsa.plot_acf(y, lags=30, alpha = 0.05)
# Because this is an AR model of lag 1, the PACF dies out after lag 1
smt.graphics.tsa.plot_pacf(y, lags=30, alpha = 0.05)

# Calculate Mean 
# Since c = 0 then this result will be 0 anyway
y_uncond_mean = c / (1 - arparams[0])
print(y_uncond_mean)

diff_y = pd.Series.diff(pd.Series(y)).dropna()

sample_mean = np.mean(y)
print(sample_mean)

#%%
# Calculate Variance
# arma_generate_sample uses np.random.randn to generate epsilon
# We know then that by default 
# var(epsilon) = 1 (or sigma^2 = 1)
# mean(epsilon) = 0

y_uncond_var = sigma2 / (1 - np.power(arparams[0],2))
print(y_uncond_var)

sample_var = np.var(diff_y)
print(sample_var)

