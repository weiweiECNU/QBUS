#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 11:34:54 2018

@author: Professor Junbin Gao

To run this program, you must update statsmodels to the newest version 0.8
"""
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import statsmodels as sm 
import statsmodels.api as smt
# We will use statsmodel's time series plots to draw ACF etc.


# Setting up the random generator 
np.random.seed(12345) 

arparams = np.array([1.0])   # please test -0.7 and 1.0
zero_lag = np.array([1])
maparams = np.array([0.6, -0.5])

ar = np.r_[1, -arparams]  # add zero-lag (coefficient 1) and negate
ma = np.r_[1, maparams]   # add zero-lag

# Mode Yt = 0.7Y_{t-1} + e_t          
y1 = sm.tsa.arima_process.arma_generate_sample(ar, zero_lag, 5000)
plt.figure()
plt.plot(y1)
smt.graphics.tsa.plot_acf(y1, lags=30, alpha = 0.05)
# Because this is an AR model of lag 1, the PACF dies out after lag 1
smt.graphics.tsa.plot_pacf(y1, lags=30, alpha = 0.05)

# Yt = e_t + 0.6e_{t-1}- 0.5 e_{t-2}
y2 = sm.tsa.arima_process.arma_generate_sample(zero_lag, ma, 250)
plt.figure()
plt.plot(y2)
smt.graphics.tsa.plot_acf(y2, lags=30, alpha = 0.05)
smt.graphics.tsa.plot_pacf(y2, lags=30, alpha = 0.05)
