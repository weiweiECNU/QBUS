#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 09:10:27 2018

@author: Professor Junbin Gao

To run this program, you must update statsmodels to the newest version 0.8
"""
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import statsmodels as sm 
import statsmodels.api as smt
# We will use statsmodel's time series plots to draw ACF etc.
from pandas.plotting import autocorrelation_plot
# Or we can use pandas to draw ACF
 
# Setting up the random generator 
np.random.seed(12345) 

data = np.random.standard_normal(1000)
 
# Plot our data
plt.plot(data)
 
# Plot the ACF for the data. This call opens a new plot
smt.graphics.tsa.plot_acf(data, lags=30, alpha = 0.05)
# lags = 30 means drawing 30 lags
# Here alpha=.05, 95% confidence intervals are returned 
# where the standard deviation is computed according to 
# Bartlett’s formula.

# You may change 0.05 to other values for alpha to see what will happen

# Plot the ACF for the data. This call opens a new plot
smt.graphics.tsa.plot_pacf(data, lags=30, alpha=0.05)

 

 