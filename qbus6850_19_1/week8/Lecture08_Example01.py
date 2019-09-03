# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 16:03:09 2018

@author: Professor Junbin Gao

My Gradient Boosting 
"""

import numpy as np
from random import shuffle
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

from sklearn import decomposition 
from sklearn import preprocessing

np.random.seed(5)

def FtoP(F):
    expF = np.exp(F)
    a = expF.sum(axis=1)
    P = expF / a[:,None]
    return P

iris = datasets.load_iris()
X = iris.data
t = iris.target
N = len(t)
index = list(range(N))
shuffle(index)

m = 12
T = 3

X1 = X[index[:m]]
t1 = t[index[:m]]
t1 = pd.get_dummies(t1)
#t1 = t1.values 

F = np.full((m,3),0.0)
rho = 1
# This is easy to check
#P = np.full((12,3),1/3)
P = FtoP(F)

NegG = t1.values - P


baseH0 = []
baseH1 = []
baseH2 = []

#First Round
for t in range(T):
    regressor = DecisionTreeRegressor(random_state=0, max_depth=2)
    h0 = regressor.fit(X1, NegG[:,0])
    baseH0.append(h0)
    F[:,0] = F[:,0] + rho*h0.predict(X1)

    regressor = DecisionTreeRegressor(random_state=0, max_depth=2)
    h1 = regressor.fit(X1, NegG[:,1])
    baseH1.append(h1)
    F[:,1] = F[:,1] + rho*h1.predict(X1)
    
    regressor = DecisionTreeRegressor(random_state=0, max_depth=2)
    h2 = regressor.fit(X1, NegG[:,2])
    baseH2.append(h2)
    F[:,2] = F[:,2] + rho*h2.predict(X1)

    #Next Round
    P = FtoP(F)
    NegG = t1.values - P
    
# Now the models are stored in BaseH0, BaseH1 and BaseH2
# Predict for a new case
x = X[index[148:150]]
F0 = 0.0
F1 = 0.0
F2 = 0.0

for t in range(T):
    F0 = F0 + baseH0[t].predict(x) 
    F1 = F1 + baseH1[t].predict(x)
    F2 = F2 + baseH2[t].predict(x) 

F = np.vstack((F0,F1,F2))
F = F.T
predictedP = FtoP(F)
print(predictedP)
