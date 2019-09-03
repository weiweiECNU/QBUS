#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 09:10:27 2018

@author: Professor Junbin Gao

To run this program, you must update statsmodels to the newest version 0.8
"""
import matplotlib.pyplot as plt 
import pandas as pd
import statsmodels as sm 
import statsmodels.api as smt
# We will use statsmodel's time series plots to draw ACF etc.
# from pandas.tools.plotting import autocorrelation_plot
from pandas.plotting import autocorrelation_plot
# Or we can use pandas to draw ACF
 
# This data is of type DataFrame with only one column
data = pd.read_csv('data.csv')
# We make it as a Series by taking out the column
data = data['Data']

# We calculate difference series, data[1]-data[0], data[2]-data[1],...
diff_data = data.diff()    
# Checking the first entry in diff_data
diff_data.iloc[0]
# Why is it a nan?
diff_data = diff_data.dropna()
# If we dont do this, what will happen for the following codes

# Plot our data
plt.plot(data)
plt.figure()
plt.plot(diff_data)

# Plot the ACF for the data. This call opens a new plot
smt.graphics.tsa.plot_acf(data, lags=30, alpha = 0.05)
# lags = 30 means drawing 30 lags
# Here alpha=.05, 95% confidence intervals are returned 
# where the standard deviation is computed according to 
# Bartlett’s formula.

# You may change 0.05 to other values for alpha to see what will happen

# Plot the ACF for the data. This call opens a new plot
# smt.graphics.tsa.plot_pacf(data, lags=30, alpha=0.05)

# For differenced time series
smt.graphics.tsa.plot_acf(diff_data, lags=30, alpha = 0.05)
#plt.figure()
#smt.graphics.tsa.plot_pacf(diff_data, lags=30, alpha = 0.05)

# Use pandas for ACF, but not plot function for PACF
plt.figure()  # We need prepare the figure
autocorrelation_plot(data)

 