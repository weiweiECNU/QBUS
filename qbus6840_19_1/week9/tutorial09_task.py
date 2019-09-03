#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:18:04 2019

@author: Boyan
"""


import matplotlib.pyplot as plt
import statsmodels.api as smt
import pandas as pd


data = pd.read_csv('sales-of-shampoo-over-a-three-ye.csv')
ts = data['Sales of shampoo over a three year period'][:-1]

plt.figure()
plt.plot(ts)
plt.title("Sales of shampoo over a three year period")

#%% take the 1st order diff, as first order diff is stationary, we then set d = 1
ts_diff = ts - ts.shift()
ts_diff.dropna(inplace=True)
plt.figure()
plt.plot(ts_diff)
plt.title("Air passenger data (log-1st diff)")

#%% p = 1, q = 1
smt.graphics.tsa.plot_acf(ts_diff, lags=20, alpha = 0.05)
smt.graphics.tsa.plot_pacf(ts_diff, lags=20, alpha = 0.05)

#%% AR(2) model
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(ts, order=(1, 1, 1))  
results = model.fit(disp=-1) 
residuals = pd.DataFrame(results.resid) 
plt.figure() 
plt.plot(residuals)
plt.title('ARIMA(1,1,1) RSS: %.4f'% sum((results.resid.values)**2))

#%%
# Get Fitted Series
fitted = results.predict(typ = 'levels', dynamic = False)

# Actual vs Fitted 
results.plot_predict(dynamic=False) 
plt.show()

