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
np.random.seed(123456)

# The ARMA process in statsmodels asks for both AR and MA order coefficients


# ARMA(4,5) process
ar = np.array([1, 0.3, 0.2, 0.1, 0.2])
ma = np.array([1, 0.5,0.4,0.3,0.5,0.2])   # statsmodel asks the coefficients on the zero-lag, which are 1s

arma_process = sm.tsa.arima_process.ArmaProcess(ar, ma)        

print("The specified process is stationary? Answer is {0}".format(arma_process.isstationary))

print("The specified process is invertible? Answer is {0}".format(arma_process.isinvertible))

# This section is to demonstrate correlation
# Simulate data for the defined the process
# burnin = 100 means we ingore the 100 initial data
y = arma_process.generate_sample(40, burnin = 100)
z1 = y.copy()
z1[:5] = np.NaN
z2 = y.copy()
z2[-5:-1] = np.NaN

z11 = y[5:] # Start from Y_3
z22 = y[:-5] # delete the last two values

plt.figure()
plt.plot(y)
plt.plot(z1, color='g')

plt.figure()
plt.plot(y)
plt.plot(z2, color='r')



plt.figure()
plt.plot(z11, color='g') 
plt.plot(z22, color='r')

 