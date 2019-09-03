 # -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 13:34:07 2017
Revised on Wed Mar  21 11:13:26 2018

@author: Professor Junbin Gao
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
 
# Load the beer dataset; data are separated by comma
beer = pd.read_csv('beer.csv')

#We are going to regress Weight variable over the Height variable
Y = beer['x'].values     # We want to predict consumption
X = [i for i in range(1,Y.size+1)]   # predictor is time
X = np.asarray(X)                    # making the list an array
 
X = X[:, np.newaxis]  # Making it one column vector, or
Y = Y[:, np.newaxis] 

X = np.double(X)      # converting integers to double type 
Y = np.double(Y)  

# Now let produce Seasonal index for the time period. We suppose 
# we start from January
seasons = []
for i in range(Y.size):
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

# If we got the seasonal frequency wrong, see what will happen.


seasons = []
for i in range(Y.size):
    if i % 4 == 0:
        seasons = np.append(seasons, 'Spring')
    if i % 4 == 1:
        seasons = np.append(seasons, 'Summer')   
    if i % 4 == 2:
        seasons = np.append(seasons, 'Autumn')
    if i % 4 == 3:
        seasons = np.append(seasons, 'Winter')
    
        
# So seasons contains categories values, we can use panda get_dummies
# We drop out category. Here April will be represented as (0,0,0,0,0,0,0,0,0,0,0)
# This is because Python orders month names in alphabetic order
dummies = pd.get_dummies(seasons, drop_first=True)

#Please note dummier is a pandas DataFrame, we shall take values out by
dummies = dummies.values

# Now we add these dummy features into feature, stacking along the column
Xnew = np.hstack((X,dummies))
        
# Create linear regression object (model)
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(Xnew, Y)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The intercept
print('Intercept: \n', regr.intercept_)

# Explained variance score: 1 is perfect prediction
print('Variance score (R^2): %.2f' % regr.score(Xnew, Y))

 
Ypred = regr.predict(Xnew) 

plt.plot(Y, label='Observed')
plt.plot(Ypred, '-r', label='Predicted')
plt.legend()
