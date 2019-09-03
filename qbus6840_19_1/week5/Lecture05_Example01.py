# -*- coding: utf-8 -*-
"""
Created on Wed Mar  28 09:14:32 2018

@author: Professor Junbin Gao
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
 
# Load the SP500 dataset;  
SP = pd.read_csv('SP500_Month.csv')

# The closing price is in coulumn five, we fetch it by making it as a pandas Series
closePrice = SP[SP.columns[4]] 


# Product time for plotting purpose
# strptime(date_string, format) returns a datetime corresponding 
# to date_string, parsed according to format
dates = SP[SP.columns[0]] 
x = np.array([dt.datetime.strptime(d, '%d/%m/%y') for d in dates]) 

# We are going to do simple exponential smoothing
# In pandas simple exponential smoothing method is implemented for pandas Series
# Let do it for four different values of alpha
smoothed_1 = closePrice.ewm(alpha = 0.05).mean()
smoothed_2 = closePrice.ewm(alpha = 0.1).mean()
smoothed_3 = closePrice.ewm(alpha = 0.6).mean()
smoothed_4 = closePrice.ewm(alpha = 0.9).mean()

#preparing the first figure
fig1 = plt.figure()
plt.plot(x,closePrice, label='Observed')
plt.plot(x,smoothed_1, '-r', label='Smoothed')
plt.title('SP500 Close Price, smoothed with alpha = 0.05') 
plt.xlabel('Year')
plt.ylabel('Close Price')
plt.legend(loc=2) 

#preparing the second figure
fig2 = plt.figure()
plt.plot(x,closePrice, label='Observed')
plt.plot(x,smoothed_2, '-r', label='Smoothed')
plt.title('SP500 Close Price, smoothed with alpha = 0.1') 
plt.xlabel('Year')
plt.ylabel('Close Price')
plt.legend(loc=2)  

#preparing the third figure
fig3 = plt.figure()
plt.plot(x,closePrice, label='Observed')
plt.plot(x,smoothed_3, '-r', label='Smoothed')
plt.title('SP500 Close Price, smoothed with alpha = 0.6') 
plt.xlabel('Year')
plt.ylabel('Close Price')
plt.legend(loc=2) 

#preparing the fourth figure
fig4 = plt.figure()
plt.plot(x,closePrice, label='Observed')
plt.plot(x,smoothed_4, '-r', label='Smoothed')
# It seems there is no smoothing effect.
plt.title('SP500 Close Price, smoothed with alpha = 0.9') 
plt.xlabel('Year')
plt.ylabel('Close Price')
plt.legend(loc=2) 