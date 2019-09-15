#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 16:00:22 2019

@author: QBUS6840 teaching team
"""

#%% Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%% Read data from a csv file
visitors = pd.read_csv('AustralianVisitors.csv')
y = visitors['No of Visitors']

#%% Manually smooth the data

# Step 1: initialize alpha and l_0
alpha = 0.1
smoothed_manual = [y[0]]

# Step 2:Using a for loop to iteratively calculate l_t 
for i in range(1,len(visitors)):
    # recall the equation l_t = alpha*y_t = (1-alpha)*l_(t-1)
    smoothed_manual.append( alpha * y[i] + (1- alpha)*smoothed_manual[i-1] )

#%% Smooth the data using Pandas. By default, pandas use set l0 to be the first 
    # value of the series
    
smoothed1 = y.ewm(alpha=0.05, adjust=False).mean()
smoothed2 = y.ewm(alpha=0.1, adjust=False).mean()
smoothed3 = y.ewm(alpha=0.3, adjust=False).mean()
smoothed4 = y.ewm(alpha=0.7, adjust=False).mean()

#%% Plot original vs smoothed series
fig = plt.figure()
plt.plot(y,label = "observations")
plt.plot(smoothed_manual, label = "manual smoothed curve, alpha = 0.1")
plt.plot(smoothed1, label = "Alpha = 0.05")
plt.plot(smoothed2, label = "Alpha = 0.1")
plt.plot(smoothed3, label = "Alpha = 0.3")
plt.plot(smoothed4, label = "Alpha = 0.7")
plt.title("Various values of Alpha")
plt.legend(loc="upper left")

# You can save the figure to your local drive
#fig.savefig("Alpha.pdf")

#%% Find the optimal value of alpha for 1-step smoothing
# Define a function to calculate sum square errors of two given series
def sse(x, y):
    return np.sum(np.power(x - y,2))

# 
sse_one = []
alphas = np.arange(0.01,1,0.01)  # Array of posible values of alpha

# For each value of alpha, obtain a smoothed series
for i in alphas:
   smoothed = y.ewm(alpha = i, adjust=False).mean()
   sse_one.append( sse(smoothed[:-1], y.values[1:]) )

#%% Plot the errors vs different values of alpha
plt.figure()
plt.plot(sse_one)
plt.title("SSE for one step smoothing")
plt.ylabel("SSE")
plt.xlabel("Alpha")
plt.xticks(np.linspace(0, 100, 11), ["{0:1.1f}".format(x) for x in np.linspace(0,1,11) ])

#%%
optimal_alpha_one = alphas[ np.argmin(sse_one) ]
print("Optimal Alpha for 1-step forecast is {0}".format(optimal_alpha_one))

#%% Find the optimal value of alpha for 2-step smoothing 
sse_two = []
alphas = np.arange(0.01,1,0.01)
for i in alphas:
    smoothed = y.ewm(alpha = i, adjust=False).mean()
    sse_two.append( sse(smoothed[:-2], y.values[2:]) )
    
#%%
plt.figure()
plt.plot(sse_two)
plt.title("SSE for two step smoothing")
plt.ylabel("SSE")
plt.xlabel("Alpha")
plt.xticks(np.linspace(0, 100, 11), ["{0:1.1f}".format(x) for x in np.linspace(0,1,11) ])

#%%
optimal_alpha_two = alphas[np.argmin(sse_two)]
print("Optimal Alpha for 2-step forecast is {0}".format(optimal_alpha_two))

#%% Trend method
# define alpha and beta in the begining
alpha = 0.5
beta = 0.4

# define initial l,b s
l = [y[0]]
b = [y[1] - y[0]]

#%%
# forecast the data 

holtsforecast_manual = []
Y = y.tolist()

# Here, we are going to forecast 12 more months after the last observation (t = 312).
for i in range(len(y)+12):
    # when we reach the end of the original data
    # then we need to forecast the t = T:T+12
    if i == len(Y):   
        Y.append(l[-1] + b[-1])
    print(i)
    l.append(alpha * Y[i] + (1 - alpha) * (l[i] + b[i])) 
    b.append(beta * (l[i+1] - l[i]) + (1 - beta) * b[i])
    # for forecasting,  Y^2 = l1 + b1
    #                   Y^3 = l2 + b2
    #                   Y^4 = l3 + b3
    #                       ...
    #                   Y^t+1 = lt+bt
    holtsforecast_manual.append(l[i] + b[i])
    
#%% if you are focusing on smoothing, then use the below code:
# Reset initial l,b
l = [y[0]]
b = [y[1] - y[0]]

holtsmoothed_manual = []
Y = y.tolist()

# Updated 12/April
# forecast the data 
for i in range(len(y)):
	l.append(alpha * Y[i] + (1 - alpha) * (l[i] + b[i]))   # Calculating l[1], l[2], etc. 
	b.append(beta * (l[i+1] - l[i]) + (1 - beta) * b[i])   # Calculating b[1], b[2], etc.
    # for smoothing,    Y^1 = l1
    #                   Y^2 = l2
    #                       ...
    #                   Y^t = lt
	holtsmoothed_manual.append(l[i+1])

#%%
plt.figure()
plt.plot(holtsforecast_manual[:], label = "holt forecast curve, alpha,beta = 0.5")
plt.plot(y, label="original data")
plt.title("Holt's linear forecasting")

#%%
plt.figure()
plt.plot(holtsmoothed_manual[:], label = "holt smoothed curve, alpha,beta = 0.5")
plt.plot(y, label="original data")
plt.title("Holt's linear smoothing")
