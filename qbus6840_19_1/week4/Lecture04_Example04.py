# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:31:09 2017
Revised on Wed Mar 21 11:35:43 2018

@author: Professor Junbin Gao
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

np.random.seed(0)
 
# Prepare data
Y = np.array([2, -2, 2, -2])

# Making a synthetic time series longer by repeating it and adding some noise
Y = np.tile(Y, 4)
Y = Y + 0.01*np.random.randn(len(Y))  # This is a one-dimensional array
Y = np.expand_dims(Y, axis=1)    # Making it to two-dimensional but only one column
                                 # This is because sklearn likes this way

# Time
T = np.expand_dims(np.arange(1,len(Y)+1),axis=1) 
             
plt.plot(T,Y)
        
# Create linear regression object (model)
regr = linear_model.LinearRegression()


# Train the model using the training sets
regr.fit(T, Y)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The intercept
print('Intercept: \n', regr.intercept_)

# Explained variance score: 1 is perfect prediction
print('Variance score (R^2): %.2f' % regr.score(T, Y))
 
Ypred = regr.predict(T) 

plt.figure()
plt.plot(T, Y, '.', color = 'black', label='Observed')
plt.plot(T, Ypred, '-r', label='Predicted')
plt.title('Synthetic Seasonal Data')
plt.legend(loc=2)  # Putting legend on the left top
 
           