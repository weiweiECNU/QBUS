#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 20:47:35 2018

@author: jbgao
"""
#%%

# Task 1
#Manually
#Generate data

from sklearn.datasets.samples_generator import    make_regression

N = 20 

x, y, true_coef = make_regression(n_samples = N, n_features = 1, noise=20, coef=True)


import matplotlib.pyplot as plt
import numpy as np


# Scatter plot
plt.figure()
plt.scatter(x, y)

#%%
'''
generate 
'''

X = np.column_stack((np.ones(N), x))

# Convert X to a matrix
X = np.asmatrix(X)

# Estimate linear regression coefficients 
lin_betas = np.linalg.inv(X.T*X) * X.T * y.reshape(N,1)

# beta_0
lin_intercept = lin_betas[0,0]
print("intercept (beta_0): {0:.2f}".format(lin_intercept))

# beta_1
lin_beta = lin_betas[1,0]
print("beta_1: {0:.2f}".format(lin_beta))

x_1 = 1

prediction = lin_intercept + lin_beta * x_1

print("Predicted value at x = {0}: {1:.2f}".format(x_1, prediction))

#%%

# Task 2
# Read data
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data
clock_auction_df = pd.read_csv("auction.txt", delimiter="\t")
#Get some info of the data
clock_auction_df.info()
clock_auction_df.head()
clock_auction_df.describe()

clock_auction_df.corr()

#%%

x_data = np.reshape(clock_auction_df["Age"].values, (len(clock_auction_df), 1)) 

y_data = np.reshape(clock_auction_df["Price"].values, (len(clock_auction_df), 1))

# Create the linear regression object
lr_obj = LinearRegression()
# Estiamte coefficients
lr_obj.fit(x_data, y_data)
# Predict the sale price of a 121 year old clock
age = 121     # new data age 121

predicted_price = lr_obj.predict(age)

print("Estimated Sale Price: ${0:.2f}".format(predicted_price[0, 0]))

#%%
# Task 3
import numpy as np
import matplotlib.pyplot as plt

"""
Build the gradient descent function

"""
# N denotes the number of training examples here, 
# not the number of features
def Gradient_Descent_Algo(x, y, beta, alpha, N, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        # predicted values from the model
        model_0 = np.dot(x, beta)
        loss_temp = model_0 - y
        # calculte the loss function
        loss = np.sum(np.square(loss_temp)) / (2 * N)
        # save all the loss function values at each step
        loss_total[i]= loss
        #You can check iteration by
        #print("Iteration: {0} | Loss fucntion: {1}".format(i, loss))
        # calcualte the gradient using matrix representation
        gradient = np.dot(xTrans, loss_temp) / N
        # update the parameters simulteneously with learning rate alpha
        beta = beta - alpha * gradient
        # save all the estimated parametes at each step
        beta_total[i,:]= beta.transpose()
    return beta

# Initialise RNG to generate the same random numbers each time
np.random.seed(0)

m = 50 #number of training examples
x = np.linspace(0.0, 1.0, m)

# Function true coefficients/parameters
beta0 = 4
beta1 = 1.5
 
# true values from linear model
f = beta0 + beta1 * x
# Add noisy
sigma2 = 0.1
y = f + np.random.normal(0, np.sqrt(sigma2), m)

# reshape
y = np.reshape(y, (len(y), 1))

x_data_1 = np.reshape(x, (len(x), 1))
x = np.column_stack((np.ones(len(x)), x_data_1))

fig0 = plt.figure()
plt.scatter(x[:,1],y)
#fig0

# using a specific number of iterations as stopping criteria
numIterations= 100000
# select the learning rate
alpha = 0.0005
loss_total= np.zeros((numIterations,1))
beta_total= np.zeros((numIterations,2))
# parameters starting values
beta_initial = np.reshape(np.zeros(2),(2, 1))
beta = Gradient_Descent_Algo(x, y, beta_initial, alpha, m, numIterations)
print(beta)

fig1 = plt.figure()
plt.plot(loss_total, label = "Loss fucntion")
plt.plot(beta_total[:,0], label = "Beta0")
plt.plot(beta_total[:,1], label = "Beta1")
plt.legend(loc="upper right")
plt.xlabel("Number of iteration")
#fig1



