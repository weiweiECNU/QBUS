# -*- coding: utf-8 -*-
"""
Created on Wed Mar  28 09:34:07 2018
Modified on Mon Mar 4, 2019

@author: Professor Junbin Gao
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import holtwinters as ht
import datetime as dt
from sklearn import linear_model


dateparse = lambda dates: pd.datetime.strptime(dates, '%b-%Y')
visitors = pd.read_csv('AustralianVisitors.csv',
parse_dates=['YearMonth'], index_col='YearMonth',
date_parser=dateparse)
print(visitors.head())

 
visitors = visitors.astype(np.double)

visitors = visitors[252:] # Look at the numbers between Jan 2012 and Dec 2016
visitors['Time'] = np.arange(1,61,1)

# Get the observed of numbers of visitors
Y =  visitors['No of Visitors']
# We will use the data between 2012 and 2015 to do decomposition
Y = Y[:48]

#preparing the first figure
fig1 = plt.figure()
plt.plot(Y, label='Observed') 
plt.title('Visitor Arrivals (2012-2015)') 
plt.xlabel('Year')
plt.ylabel('Number of Visitors')
plt.legend(loc=2) 
fig1.savefig('2019Lecture03_F01.png')

# We do rolling window 12 then window 2 to simulate CMA-12  
TC = Y.rolling(2, center = True).mean().rolling(12, center = True).mean() 
# Think about why we do this
TC = TC.shift(-1)

# Display the smoothed against the original time series
fig2 = plt.figure()
plt.plot(Y, label='Observed')
plt.plot(TC, '-r', label='Smoothed')
plt.title('Visitor Arrivals (2012-2015)') 
plt.xlabel('Year')
plt.ylabel('Number of Visitors')
plt.legend(loc=2) 
fig2.savefig('2019Lecture03_F02.png')

# Display the smoothed only for better view
# The smoothed is the initial estimate of Trend-Cycle components
fig3 = plt.figure() 
plt.plot(TC, '-r', label='Smoothed')
plt.title('Visitor Arrivals (2012-2015)') 
plt.xlabel('Year')
plt.ylabel('Number of Visitors')
plt.legend(loc=2) 
fig3.savefig('2019Lecture03_F03.png')

# Calculate the approximated seasonal component by removing TC components
# If it is additive model, we shall use vs - r
Shat = Y / TC

# Showing the reasonable component
fig4 = plt.figure()
plt.plot(Shat, '-r', label='Seasonal Approximation')
plt.title('Approximated Seasonal Component (2012-2015)') 
plt.xlabel('Year')
plt.ylabel('Number of Visitors')
plt.legend(loc=2) 
fig4.savefig('2019Lecture03_F04.png')

# We organize four years of data in a DataFrame for convenient calculation
d = {'year1': Shat.values[6:18], 'year2': Shat.values[18:30], 'year3': Shat.values[30:42]}
df = pd.DataFrame(data=d, index = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])

# Taking average by month for four years data
df['Average'] = (df['year1'] + df['year2'] + df['year3'])/3.0

# Calculating normalisation factor
c = 12.0/df['Average'].sum()   # c = 1.00026

# Doing normalisation gives us the seasonal index
df['Normalized'] = df['Average'] * c

# Make a seasonal time series with all the indices, the time series is repeated 12 seasonal index values
Shat.values[0:6] = df['Normalized'].values[6:12]
Shat.values[6:18] = df['Normalized'].values
Shat.values[18:30] = df['Normalized'].values
Shat.values[30:42] = df['Normalized'].values
Shat.values[42:48] = df['Normalized'].values[0:6] 

# Getting the seasonaly adjusted time series
vss = pd.DataFrame(Y)
vss['Adjusted'] = vss['No of Visitors'] / Shat

fig5 = plt.figure()
plt.plot(vss['Adjusted'], label='Adjusted') 
plt.title('Adjusted Visitor Arrivals (2012-2015)') 
plt.xlabel('Year')
plt.ylabel('Number of Visitors')
plt.legend(loc=2) 
fig5.savefig('2019Lecture03_F08.png')

fig6 = plt.figure()
plt.plot(Y, label='Observed')
plt.plot(vss['Adjusted'], label='Adjusted') 
plt.title('Adjusted Visitor Arrivals (2012-2015)') 
plt.xlabel('Year')
plt.ylabel('Number of Visitors')
plt.legend(loc=2)  
fig6.savefig('2019Lecture03_F09.png')

# Do linear regression for modeling trend-cycle component
lm = linear_model.LinearRegression(fit_intercept=True)
T = visitors['Time'].values
T0 = T[:48]
 # Fitting linear model to data
model = lm.fit(T0.reshape(-1,1),vss['Adjusted'])   # We need convert to 2D form requested by sklearn
# Checking the model parameter values
print(model.coef_)    #3261.32994967
print(model.intercept_)  #476759.2518553154

# To draw the regression line, calculate estimate at the time points 1 to 48
vss['Regressed'] = lm.predict(T0.reshape(-1,1)) 

# Showing regression model
fig7 = plt.figure()
plt.plot(vss['Regressed'], label='Trend-Cycle Model')
plt.plot(vss['Adjusted'], label='Adjusted') 
plt.title('The Trend (and Cycle) (2012-2015)') 
plt.xlabel('Year')
plt.ylabel('Number of Visitors')
plt.legend(loc=2)  
fig7.savefig('2019Lecture03_F10.png')

# Errors
e = vss['No of Visitors'] /(Shat * vss['Regressed'])
fig7a = plt.figure()
plt.plot(e, label='Errors')  
plt.title('Errors (2012-2015)') 
plt.xlabel('Year')
plt.ylabel('Errors')
plt.legend(loc=2)  
plt.axhline(y = 1.0, color='black')
fig7a.savefig('2019Lecture03_F11.png')

# Smoothing the error to get Cycle-Error
eSmooth = e.rolling(3, center = True).mean()
fig7b = plt.figure()
plt.plot(eSmooth, label='Smoothed Errors')  
plt.title('Cycle and Errors (2012-2015)') 
plt.xlabel('Year')
plt.ylabel('Cycle Values')
plt.legend(loc=2)  
plt.axhline(y = 1.0, color='black')
fig7b.savefig('2019Lecture03_F12.png')

# Final Errors
eFine = vss['No of Visitors'] /(Shat * vss['Regressed']* eSmooth)
fig7c = plt.figure()
plt.plot(eFine, label='Final Errors')  
plt.title('Final Errors (2012-2015)') 
plt.xlabel('Year')
plt.ylabel('Errors')
plt.legend(loc=2)  
plt.axhline(y = 1.0, color='black')
fig7c.savefig('2019Lecture03_F13.png')
 
# We are doing forecasting for Feb and May 2016
# These months are counted as the 50th and 53th months in time
t1 = np.array([50, 53])    # The time for Feb and May 2016

# Trend components of Feb and May 2016
TC = lm.predict(t1.reshape(-1,1))  # times series forecasting
# The seasonal index for Feb
SI = Shat.values[1]
# The forecast for Feb 2018
PredictedFeb2017 = TC[0]*SI   #  = 743273
# The seasonal index for March
SI = Shat.values[4]
# The forecast for Feb 2018
PredictedMay2017 = TC[1]*SI   # = 518879

# The Model Estimated for 2016
visitors['Regressed'] = lm.predict(T.reshape(-1,1)) 

fig8 = plt.figure()
plt.plot(visitors['No of Visitors'], label='Observed')
plt.plot(visitors['Regressed'], label='Linear Trend') 
plt.title('Trend (and Cycle) (2012-2016)') 
plt.xlabel('Year')
plt.ylabel('Number of Visitors')
plt.legend(loc=2)  
local_date = pd.to_datetime('2015-12-01')
plt.axvline(x=local_date, color='red') 
fig8.savefig('2019Lecture03_F14.png')

# Extend Seasonal Component to 2016 
visitors['Seasons']=visitors['No of Visitors']
SS = visitors['Seasons']
SS.values[0:48] = Shat.values
SS.values[48:60] = Shat.values[0:12]

fig9 = plt.figure() 
plt.plot(SS, label='Seasonal Component') 
plt.title('Seasonal Component to Future (2012-2016)') 
plt.xlabel('Year')
plt.ylabel('Number of Visitors')
plt.legend(loc=2)  
local_date = pd.to_datetime('2015-12-01')
plt.axvline(x=local_date, color='red')  
fig9.savefig('2019Lecture03_F15.png')

# All forecasts for 2016
t1 = np.array([49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]) 
TC = lm.predict(t1.reshape(-1,1)) 
Forecasts = TC * Shat.values[0:12]

visitors['Forecasts']=visitors['No of Visitors']
SS = visitors['Forecasts']
SS.values[0:48] = np.nan
SS.values[48:60] = Forecasts

fig10 = plt.figure() 
plt.plot(SS, label='Forecasts') 
plt.plot(visitors['No of Visitors'], label='Observed') 
plt.title('Forecasts for 2016') 
plt.xlabel('Year')
plt.ylabel('Number of Visitors')
plt.legend(loc=2)  
local_date = pd.to_datetime('2015-12-01')
plt.axvline(x=local_date, color='red') 
fig10.savefig('2019Lecture03_F16.png')

