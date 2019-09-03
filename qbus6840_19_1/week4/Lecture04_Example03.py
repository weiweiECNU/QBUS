# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 13:34:07 2017
Revised on Wed Mar  21 11:21:31 2018

@author: Professor Junbin Gao
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
 
# Load the population dataset; data are separated by whitespace
populations = pd.read_csv('Population.csv', delim_whitespace = True)

#We are going to regress Weight variable over the Height variable
Y = np.double(populations['Population(M)'].values)     # We want to predict consumption
Xyr = np.double(populations['Year'].values)        # Year 
Xyrsq = Xyr * Xyr                                  # Year squared
 
Xyr= Xyr[:, np.newaxis]  # Making it one column vector, or
Y = Y[:, np.newaxis] 
Xyrsq= Xyrsq[:, np.newaxis]

# Now we make predictors of year and squared year
Xnew = np.hstack((Xyr, Xyrsq))
        
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

plt.plot(Xyr, Y, '.', color = 'black', label='Observed')
plt.plot(Xyr, Ypred, '-r', label='Predicted')
plt.title('American Populations (1790-2010)')
plt.xlim([1790, 2010])
plt.legend(loc=2)  # Putting legend on the left top