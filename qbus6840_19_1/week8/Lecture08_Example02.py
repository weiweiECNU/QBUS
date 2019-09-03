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
np.random.seed(12345)

# The ARMA process in statsmodels asks for both AR and MA order coefficients


# ARMA(4,5) process
ar = np.array([1, 0.3, 0.2, 0.1, 0.2])
ma = np.array([1, 0.5,0.4,0.3,0.5,0.2])   # statsmodel asks the coefficients on the zero-lag, which are 1s

arma_process = sm.tsa.arima_process.ArmaProcess(ar, ma)        

print("The specified process is stationary? Answer is {0}".format(arma_process.isstationary))

print("The specified process is invertible? Answer is {0}".format(arma_process.isinvertible))

# Simulate data for the defined the process
# burnin = 100 means we ingore the 100 initial data
y = arma_process.generate_sample(250, burnin = 100)

plt.figure()
plt.plot(y)
smt.graphics.tsa.plot_acf(y, lags=30, alpha = 0.05)
# Because this is an MA model of lag 2, the ACF dies out after lag 2
smt.graphics.tsa.plot_pacf(y, lags=30, alpha = 0.05)

 