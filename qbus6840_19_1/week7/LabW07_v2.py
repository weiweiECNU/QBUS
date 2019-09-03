#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:04:05 2017

@author: Boyan Zhang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

visitors = pd.read_csv('AustralianVisitors.csv')
y = visitors['No of Visitors']

#%%
from math import sqrt
from sklearn.linear_model import LinearRegression

# optimize the best fitting l0, bo, and s
def linearOptimization(X, m):
	# define X: 1~t
	x = np.linspace(1, len(X), len(X))
	x = np.reshape(x, (len(x), 1))
	y = np.reshape(X, (len(X), 1))
	# train a linear regression to get l0 (intercept_) and b0 (coef_[0])
	lm = LinearRegression().fit(x, y)
	l0 = lm.intercept_
	b0 = lm.coef_[0]
    
	# use trained linear regression model to get y^t
	# then s^ = y-y^
	# finally average s^ to get s
	# The following statement is equal to:
	#    res = y - lm.predict(x)+0.
	#    res = np.reshape(res,(m,int(len(X)/m)))
	#    s = np.mean(res,axis=0)
	s = np.mean(np.reshape(y + 0. - lm.predict(x),(int(len(X)/m),m)), axis=0)
	# in the above statement, 0. is used to transfer a int to float

	# check the datatype of l0, b, and s
	return l0[0],b0, s.tolist()

# additive Holt Winters
def addSeasonal(x, m, fc, alpha = None, beta = None, gamma = None, \
                l0 = None, b0 = None, s = None):
    # initial some temp variables
	Y = x[:]
	l = []
	b = []
	s = []
	# if hyper paras have not defined 
	if (alpha == None or beta == None or gamma == None):
		alpha, beta, gamma = 0.1, 0.1, 0.1
	# if l0 b0 and s have not initialized
	if (l0 == None or b0 == None or s == None):
		l0,b0,s = linearOptimization(Y,m)
		l.append(l0)
		b.append(b0)

	else:
		l = l0
		b = b0
		s = s

	forecasted = []
	rmse = 0
    
	# forecast from T = 1:t+fc
	for i in range(len(x) + fc):
		# when we reach the end of the original data
		# then we need to forecast the t = T:T+12
		if i == len(Y) :   
			Y.append(l[-1] + b[-1] + s[-m])
		# update the l,b,s and y
		# lt = alpha * (yt - St-m) + (1-alpha) * (lt-1 + bt-1)
		l.append(alpha * (Y[i] - s[i-m]) + (1 - alpha) * (l[i] + b[i])) 
 		# bt = beta * (lt - lt-1) + (1-beta) * bt-1   
		b.append(beta * (l[i + 1] - l[i]) + (1 - beta) * b[i])
 		# St = gamma * (yt - lt-1 - bt-1) + (1-gamma) * St-m  
		s.append(gamma * (Y[i] - l[i] - b[i]) + (1 - gamma) * s[i-m]) 
		# for forecasting, Yt+1 = lt +bt + st+1-m
		forecasted.append(l[i] + b[i] + s[i-m])
        
	# calculate the rmse
	# The zip() function take iterables (can be zero or more), 
	# makes iterator that aggregates elements based on the iterables passed, 
	# and returns an iterator of tuples.
	rmse = sqrt(sum([(m - n + 0.) ** 2 for m, n in zip(Y[:-fc], y[:-fc - 1])]) / len(Y[:-fc]))
	return forecasted, Y[-fc:], l, b, s, rmse

#%%
fc = 12   # One year forecasting
M = 12    # seasonal number 

# check out the y datatype here
s_smoothed, Ys, l, b, s, rmse = addSeasonal(x = y.tolist(), m = M, fc = fc)

#%%
plt.figure(figsize=(15,8))
plt.plot(y, label='Observed')
plt.plot(s_smoothed[:], '-r', label='Forecasting result')
plt.title('Visitor Number') 
plt.xlabel('Month')
plt.ylabel('Number of Visitors')
plt.legend(loc=2)  

#%%
# remove l0,b0,s-12
states = pd.DataFrame(np.c_[l[1:], b[1:], s[12:]], \
                       columns=['level','slope','seasonal'])
fig, [ax1,ax2,ax3] = plt.subplots(3, 1, figsize=(15,8))
states['level'].plot(ax=ax1)
states['slope'].plot(ax=ax2)
states['seasonal'].plot(ax=ax3)
plt.show()

#%%
# set seasonal as additive or multiplicative
# for more details please refer: 
# https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
fit1 = ExponentialSmoothing(y, seasonal_periods=12, trend='add', seasonal='add').fit()
fit2 = ExponentialSmoothing(y, seasonal_periods=12, trend='add', seasonal='mul').fit()

#%%
# symbol r $ and \ in the results variable are the latex symbols for visualization in notebook
results = pd.DataFrame(index=[r"$\alpha$",\
                            r"$\beta$",\
                            r"$\phi$",\
                            r"$\gamma$",\
                            r"$l_0$",\
                            "$b_0$",\
                            "SSE"])
# ExponentialSmoothing() object has following attributes
params = ['smoothing_level', \
          'smoothing_slope', \
          'damping_slope', \
          'smoothing_seasonal', \
          'initial_level', \
          'initial_slope']

# check out the performance of additive and multiplicative
results["Additive"]       = [fit1.params[p] for p in params] + [fit1.sse]
results["Multiplicative"] = [fit2.params[p] for p in params] + [fit2.sse]

#%%
plt.figure()
ax = y.plot(figsize=(15,6), color='black', title="Forecasts from Holt-Winters' multiplicative method" )
ax.set_ylabel("Number of Australia Vistors")
ax.set_xlabel("Year")

# transfer the datatype to values
smooth_add = fit1.fittedvalues
smooth_mul = fit2.fittedvalues
smooth_add.plot(ax=ax, style='-', color='red')
smooth_mul.plot(ax=ax, style='--', color='green')

#%%
# forecast 12 more data and plot
forecast1 = fit1.forecast(12).rename('Holt-Winters (add-add-seasonal)')
forecast1.plot(ax=ax, style='-', color='red', legend=True)
fit2.forecast(12).rename('Holt-Winters (add-mul-seasonal)').plot(ax=ax, style='--', marker='o', color='green', legend=True)

# sometimes you may need to call .show() function to display the plots
plt.show()

#%%
print("Forecasting Australia visitors using Holt-Winters method with \n \
      both additive and multiplicative seasonality.")
print(results)
#%%
# compare the forecasting outcomes of additive and multiplicative
# np.c_: Translates slice objects to concatenation along the second axis.
df1 = pd.DataFrame(np.c_[y, fit1.level, fit1.slope, fit1.season, fit1.fittedvalues], 
                  columns=[r'$y_t$',r'$l_t$',r'$b_t$',r'$s_t$',r'$\hat{y}_t$'],index=y.index)
df1.append(fit1.forecast(24).rename(r'$\hat{y}_t$').to_frame())
#%%
df2 = pd.DataFrame(np.c_[y, fit2.level, fit2.slope, fit2.season, fit2.fittedvalues], 
                  columns=[r'$y_t$',r'$l_t$',r'$b_t$',r'$s_t$',r'$\hat{y}_t$'],index=y.index)
df2.append(fit2.forecast(24).rename(r'$\hat{y}_t$').to_frame())
#%%
# Ploting the level, trend and season for fit1 and fit2
# define 2 states variable for conveinence
states_add = pd.DataFrame(np.c_[fit1.level, fit1.slope, fit1.season], \
                       columns=['level','slope','seasonal'], \
                       index=y.index)
states_mul = pd.DataFrame(np.c_[fit2.level, fit2.slope, fit2.season], \
                       columns=['level','slope','seasonal'], \
                       index=y.index)

# define subplots windows
fig, [[ax1, ax4],[ax2, ax5], [ax3, ax6]] = plt.subplots(3, 2, figsize=(12,8))
states_add['level'].plot(ax=ax1)
states_add['slope'].plot(ax=ax2)
states_add['seasonal'].plot(ax=ax3)
states_mul['level'].plot(ax=ax4)
states_mul['slope'].plot(ax=ax5)
states_mul['seasonal'].plot(ax=ax6)
plt.show()

