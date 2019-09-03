# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 09:34:07 2017

@author: Professor Junbin Gao
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import holtwinters as ht
import datetime as dt


# The first part repeats Lecture05_Example02.py for comparison
 
# Load the Australian Visitor dataset;  
visitors = pd.read_csv('AustralianVisitors.csv')

# The closing price is in coulumn five, we fetch it by making it as a pandas Series
numVisitors = visitors[visitors.columns[1]]
numVisitors = numVisitors[228:]    # Look at the numbers between Jan 2010 and Dec 2016
months = visitors[visitors.columns[0]]
months = months[228:]

# We are going to do simple exponential smoothing
# In pandas simple exponential smoothing method is implemented for pandas Series
# Let do it for four different values of alpha
smoothed_1 = numVisitors.ewm(alpha = 0.1).mean()
smoothed_2 = numVisitors.ewm(alpha = 0.5).mean()

# Product time for plotting purpose
x = np.array([dt.datetime.strptime(d, '%b-%Y') for d in months]) 
# Prediction Year
xp = np.array([dt.datetime.strptime(d, '%b-%Y') for d in ('Jan-2017','Feb-2017',
    'Mar-2017','Apr-2017','May-2017','Jun-2017','Jul-2017','Aug-2017','Sep-2017',
    'Oct-2017','Nov-2017','Dec-2017')]) 
xp = np.hstack((x,xp))

#preparing the first figure
fig1 = plt.figure()
plt.plot(x,numVisitors, label='Observed')
plt.plot(x,smoothed_1, '-r', label='Smoothed')
plt.title('Visitor Arrivals (2010-2016), smoothed with alpha = 0.1') 
plt.xlabel('Year')
plt.ylabel('Number of Visitors')
plt.legend(loc=2) 

#preparing the second figure
fig2 = plt.figure()
plt.plot(x,numVisitors, label='Observed')
plt.plot(x,smoothed_2, '-r', label='Smoothed')
plt.title('Visitor Arrivals (2010-2016), smoothed with alpha = 0.5') 
plt.xlabel('Year')
plt.ylabel('Number of Visitors')
plt.legend(loc=2) 

# Now we try to use the trend corrected exponential smoothing which is also known 
# the Holt's linear exponential smoothing  
# Unfortunately pandas does not implement this smoothing method.  Here we are going 
# to use a third party's implementation, which is the linear method in holtwinters.py

# Preparing data
# numVisitors is a pandas Series. As holtwinters.py only accepts a list 
# variable, we need convert a Series to a python list
ts = numVisitors.tolist()

# Now we define how many predictions we wish to predict
fc = 12   # One year forecasting

# Let us try the Holt's linear exponential smoothing
t_smoothed, Y, alpha, gamma, rmse = ht.linear(ts, fc)
# The output Y contains the fc predictions
# alpha, beta are the optimal parameter values found by the program
# rmse is the RMSE of the one-step prediction

#preparing the second figure
fig3 = plt.figure()
plt.plot(x,ts, label='Observed')
plt.plot(xp,t_smoothed[:-1], '-r', label='Smoothed')
plt.title('Visitor Arrivals (2010-2016): alpha = ' + str(np.round(alpha,2)) + ', gamma = ' + str(np.round(gamma,2))) 
plt.xlabel('Year')
plt.ylabel('Number of Visitors')
plt.legend(loc=2)  

# The second part
# Now we define how many predictions we wish to predict
fc = 12   # One year forecasting
M = 12    # Number of seasons 
# Let us try the Holt-Winters's Additive Method, 
# We are not going to define alpha, beta and gamma and the program will find the best
s_smoothed, Ys, s, alpha, beta, gamma, rmse = ht.additive(ts, M, fc, alpha = None, beta = None, gamma = None)
# The output Ys contains the fc predictions
# alpha, beta, gamma are the optimal parameter values found by the program
# rmse is the RMSE of the one-step prediction
#preparing the second figure
fig4 = plt.figure()
plt.plot(x,ts, label='Observed')
plt.plot(xp,s_smoothed[:-1], '-r', label='Smoothed')
plt.title('Visitor Arrivals (2010-2016): alpha = ' + str(np.round(alpha,2)) + ', beta = ' + str(np.round(beta,2)) + ', gamma = ' + str(np.round(gamma,2))) 
plt.xlabel('Year')
plt.ylabel('Number of Visitors')
plt.legend(loc=2)  

fig5 = plt.figure()
plt.plot(s)
plt.title('Estimated Seasonal Components')