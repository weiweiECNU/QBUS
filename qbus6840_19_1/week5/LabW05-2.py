#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:35:32 2019

@author: Boyan Zhang
Updated by: Mingyuan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#%%
beer_df = pd.read_csv('beer.txt')

#%%
X = np.linspace(1, len(beer_df), len(beer_df))
y = beer_df['Sales']

plt.figure()
plt.plot(X, y)
plt.title("Beer Sales") 

#%%
X = np.reshape(X, (len(beer_df), 1))
#y = np.reshape(y, (len(beer_df), 1))
#y = y.values.reshape(len(beer_df),1)
#%%
lm = LinearRegression()
#%%
lm.fit(X, y)
#lm.predict(X_test)
#%%
# The coefficients
print("Coefficients: {0}".format(lm.coef_))
# .format(): transfer a float or an interger (int) to a string (for a word)
# The intercept
print("Intercept: {0}".format(lm.intercept_))
print("Total model: y = {0} + {1} X".format(lm.intercept_, lm.coef_[0]))
print("Variance score (R^2): {0:.3f}".format(lm.score(X, y)))
#{0},{1},{2},etc: the first position, the second position, the third position,etc
# {0:.nf}: the floats with n decimal points
#%%
trend = lm.predict(X)
#%%
#given the values X, trend is the y's from lm function: y = 155.40974026 + -0.21425154 X
plt.figure()
plt.plot(X, y, label = "Beer Sales")
plt.plot(trend, label="Trend")
plt.legend()
plt.title("Beer Sales Trend from Linear Regression Model")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()

#%%
sse1 = np.sum( np.power(y - trend,2), axis=0)
#along the first dimension (columns), calculate (y-trend)^2
sse2 = lm._residues
#sum of squared residuals
#%%
beer_df['residuals'] = y - trend
#%%
acf_vals = [beer_df['residuals'].autocorr(i) for i in range(1,25) ]
# calculate the autocorrelation between the residuals across 1 to 15 lags
# i.e., e_t and e_{t+1}, e_{t} and e_{t+2},...,e_{t} and e_{t+15}
#%%
plt.figure()
plt.bar(np.arange(1,25), acf_vals)
plt.title("ACF of Beer Sales")
plt.xlabel("Month delay/lag")
plt.ylabel("Correlation Score")
plt.show(block=False)

#%%
plt.figure()
plt.title("Residual plot")
plt.scatter(X, trend - y)
plt.xlabel("Month")
plt.ylabel("Residuals")
plt.show(block=False)
# it looks that there is no relationship between any data points: the residuals are 
# independent to each other
#%%
data = np.arange(1, 72)
forecast = lm.predict(np.reshape(np.arange(72), (72,1)))
# forecast 16 steps ahead, since predict on 72 time steps from the initial time period
# t=x=0 until the 72nd step
#%%
plt.figure()
plt.plot(X, y, label="Beer Sales")
plt.plot(trend, label="Trend")
plt.plot(forecast, linestyle='--', label="Forecast")
plt.legend()
plt.title("Beer Sales Forecast from Trend Only Linear Regression Model")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show(block=False)

#%%Find the seasonal index
T = beer_df.rolling(12, center = True).mean().rolling(2, center = True).mean().shift(-1)

S_additive = beer_df['Sales'] - T['Sales']
#%%
safe_S = np.nan_to_num(S_additive)
monthly_S = np.reshape(np.concatenate((safe_S, [0,0,0,0]), axis = 0), (5,12))
# we want to form the seasonal index for 5 years, so we add on 4 extra points 
#%%
monthly_avg = np.mean(monthly_S[1:4,], axis=0)
#ignore the ones with missing values
mean_allmonth = monthly_avg.mean()
monthly_avg_normed = monthly_avg - mean_allmonth

tiled_avg = np.tile(monthly_avg_normed, 6)

#lm.fit(X[7:50], T['Sales'].iloc[7:50])
#use the data which are not missing values to re-estimate the trend
linear_trend = lm.predict(np.reshape(np.arange(1, 72), (72,1)))
linear_seasonal_forecast = linear_trend + tiled_avg

plt.figure()
plt.plot(X, y, label="Original Data")
plt.plot(linear_trend, label="Linear Model trend")
plt.plot(linear_seasonal_forecast, label="Linear+Seasonal Forecast")
plt.title("Beer Sales Forecast from Trend+Seasonal Linear Regression Model")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.show(block=False)

#%%
# we start from January
seasons = []
for i in range(y.size):
    if i % 12 == 0:
        seasons = np.append(seasons, 'Jan')
    if i % 12 == 1:
        seasons = np.append(seasons, 'Feb')   
    if i % 12 == 2:
        seasons = np.append(seasons, 'Mar')
    if i % 12 == 3:
        seasons = np.append(seasons, 'Apr')
    if i % 12 == 4:
        seasons = np.append(seasons, 'May')
    if i % 12 == 5:
        seasons = np.append(seasons, 'Jun')   
    if i % 12 == 6:
        seasons = np.append(seasons, 'Jul')
    if i % 12 == 7:
        seasons = np.append(seasons, 'Aug')
    if i % 12 == 8:
        seasons = np.append(seasons, 'Sep')
    if i % 12 == 9:
        seasons = np.append(seasons, 'Oct')   
    if i % 12 == 10:
        seasons = np.append(seasons, 'Nov')
    if i % 12 == 11:
        seasons = np.append(seasons, 'Dec')
        

#%%
#dummies1 = pd.get_dummies(seasons)
dummies = pd.get_dummies(seasons, drop_first=True)
#Whether to get k-1 dummies out of k categorical levels by removing the first level 
# In this case, we treat 
# b0 + b1 * t as April
# b0 + b1 * t + b2 * x1 as August
# b0 + b1 * t + b3 * x2 as Dec, etc...
# please refer to the dummies column name
#%%
#Please note dummier is a pandas DataFrame, we shall take values out by
dummies = dummies.values

#%%
# make sure the size of X and dummies match
# If you are using numpy version below than 1.10, you need to uncomment the following statement
# X = X[:, np.newaxis]

#[:,np.newaxis]: make it as column vector by inserting an axis along second dimension
# Now we add these dummy features into feature, stacking along the column
Xnew = np.hstack((X,dummies))
#%%
# Create linear regression object (model)
regr = LinearRegression()

# Train the model using the training sets
regr.fit(Xnew,y)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The intercept
print('Intercept: \n', regr.intercept_)


#%%
Ypred = regr.predict(Xnew) 
plt.figure()
plt.plot(y, label='Observed')
plt.plot(Ypred, '-r', label='Predicted')
plt.legend()
#plt.close("all") #close all previous figures 

#%% Remove Jan
dummies1 = pd.get_dummies(seasons)
#%%
dummies1 = dummies1.drop(['Jan'], axis=1)
#%%
#dummies1 = dummies1.values
Xnew1 = np.hstack((X,dummies1))
regr1 = LinearRegression()
regr1.fit(Xnew1,y)
Ypred1 = regr.predict(Xnew1) 
# The coefficients
print('Coefficients: \n', regr1.coef_)
# The intercept
print('Intercept: \n', regr1.intercept_)
plt.plot(Ypred1, '-g', label='Predicted new')
plt.legend()

#%% Do not remove anything
dummies2 = pd.get_dummies(seasons)
#dummies1 = dummies1.drop(['Jan'], axis=1)
dummies2 = dummies2.values
Xnew2 = np.hstack((X,dummies2))
#%%
regr2 = LinearRegression()
regr2.fit(Xnew2,y)
Ypred2 = regr2.predict(Xnew2) 
#%%
# The coefficients
print('Coefficients: \n', regr2.coef_)
# The intercept
print('Intercept: \n', regr2.intercept_)
plt.plot(Ypred2, '-b', label='Predicted new 2')
plt.legend()
