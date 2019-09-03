# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 12:21:22 2018

@author: Professor Junbin Gao

"""
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import statsmodels as sm 
import statsmodels.api as smt

# Make sure we can replicate the results
# np.random.seed(12345)

# The ARMA process in statsmodels asks for both AR and MA order 
# coefficients

# MA(1) process
ar = np.array([1])
ma = np.array([1, .4])   # statsmodel asks the coefficients on the zero-lag, which are 1s

"""
# Produce the process
# y_t =  e_t + 0.4 e_{t-1}
"""
ma_process = sm.tsa.arima_process.ArmaProcess(ar, ma)        

print("The specified process is stationary? Answer is {0}".format(ma_process.isstationary))

print("The specified process is invertible? Answer is {0}".format(ma_process.isinvertible))

# Simulate data for the defined the process
# burnin = 100 means we ingore the 100 initial data
y1 = ma_process.generate_sample(250, burnin = 100)

plt.figure()
plt.plot(y1)
smt.graphics.tsa.plot_acf(y1, lags=30, alpha = 0.05)
# Call plot_acf from statsmodels.api.graphics.tsa
#sm.tsa.api.graphics.plot_acf(y1, lags=30, alpha = 0.05)
# Because this is an MA model of lag 1, the ACF dies out after lag 1
smt.graphics.tsa.plot_pacf(y1, lags=30, alpha = 0.05)


# MA(2) process
ar = np.array([1])
ma = np.array([1, 0.5, 0.4])   # statsmodel asks the coefficients on the zero-lag, which are 1s

ma2_process = sm.tsa.arima_process.ArmaProcess(ar, ma)        

print("The specified process is stationary? Answer is {0}".format(ma2_process.isstationary))

print("The specified process is invertible? Answer is {0}".format(ma2_process.isinvertible))

# Simulate data for the defined the process
# burnin = 100 means we ingore the 100 initial data
y2 = ma2_process.generate_sample(250, burnin = 100)

plt.figure()
plt.plot(y2)
smt.graphics.tsa.plot_acf(y2, lags=30, alpha = 0.05)
# Because this is an MA model of lag 2, the ACF dies out after lag 2
smt.graphics.tsa.plot_pacf(y2, lags=30, alpha = 0.05)



# MA(5) process
ar = np.array([1])
ma = np.array([1, 0.7,0.4,0.3,0.5,0.1])   # statsmodel asks the coefficients on the zero-lag, which are 1s

ma5_process = sm.tsa.arima_process.ArmaProcess(ar, ma)        

print("The specified process is stationary? Answer is {0}".format(ma5_process.isstationary))

print("The specified process is invertible? Answer is {0}".format(ma5_process.isinvertible))

# Simulate data for the defined the process
# burnin = 100 means we ingore the 100 initial data
y5 = ma5_process.generate_sample(250, burnin = 100)

plt.figure()
plt.plot(y5)
smt.graphics.tsa.plot_acf(y5, lags=30, alpha = 0.05)
# Because this is an MA model of lag 2, the ACF dies out after lag 2
smt.graphics.tsa.plot_pacf(y5, lags=30, alpha = 0.05)