# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 13:34:07 2017
Revised on Wed Mar  21 11:04:43 2018

@author: Professor Junbin Gao
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
 
# Load the class dataset; data are separated by whitespace
students = pd.read_csv('class.csv', delim_whitespace = True)

#We are going to regress Weight variable over the Height variable
X = students['Height'].values
Y = students['Weight'].values
# Note: the above X and Y is one dimensional array. As all the sklearn functions
# assume data are in matrix shape (row number correponds to the number of train data 
# while column number corresponds to the number of predictors (or features))

# Two ways to change the shape
X = X[:, np.newaxis]  # Making it one column vector, or
Y = np.reshape(Y,(Y.size,1))

# Create linear regression object (model)
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X, Y)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The intercept
print('Coefficients: \n', regr.intercept_)

# Explained variance score: 1 is perfect prediction
print('Variance score (R^2): %.2f' % regr.score(X, Y))

# Plot outputs
plt.scatter(X, Y,  color='black')
plt.plot(X, regr.predict(X), color='blue', linewidth=3)

 