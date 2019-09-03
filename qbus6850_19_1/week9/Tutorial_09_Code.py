#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 16:19:18 2018

@author: Boyan
"""
#%% 
# Task 1 Gradient Boosting Decistion Tree for Classification

import numpy as np
import pandas as pd
from random import shuffle
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor

np.random.seed(5)

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

#%%
F = np.full((m,3),0.0)
rho = 1
# This is easy to check
#P = np.full((12,3),1/3)

def FtoP(F):
    expF = np.exp(F)
    a = expF.sum(axis=1)
    P = expF / a[:,None]
    return P

P = FtoP(F)

#%%
NegG = t1.values - P

#%%
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

#%%
#Next Round
P = FtoP(F)
NegG = t1.values - P
    
#%%
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

#%%
F = np.vstack((F0,F1,F2))
F = F.T
predictedP = FtoP(F)
print(predictedP)

#%%

# # Task 2: Predict Onset of Diabetes (Classification)
# ## Step1: Load and Prepare Data
import numpy as np
import pandas as pd
from numpy import loadtxt

from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[4]:
# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")


# In[5]:
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]

# In[6]:
# split data into train and test sets
seed = 7
test_size = 0.33
#test_size = 33
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    Y, 
                                                    test_size=test_size, 
                                                    random_state=seed)


#%% Step 2: Train the XGBoost Model
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)

# In[8]:
print(model)

# In[9]:
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]


# In[10]:
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#%%
# # Task 3: Boston Housing (regression)
#
# ## Step1: Load and Prepare Data
import xgboost as xgb

import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.datasets import load_boston

rng = np.random.RandomState(31337)

# In[26]:
print("Boston Housing: regression")
boston = load_boston()

# In[27]:
y = boston['target']
X = boston['data']


# In[28]:
print(X.shape)


#%% Step 2: Applying k-fold Cross Validation
kf = KFold(n_splits=2, shuffle=True, random_state=rng)


# In[30]:
for train_index, test_index in kf.split(X):
    xgb_model = xgb.XGBRegressor().fit(X[train_index], y[train_index])
    predictions = xgb_model.predict(X[test_index])
    actuals = y[test_index]
    print(mean_squared_error(actuals, predictions))


# %% Step 4: Parameter Optimization
xgb_model = xgb.XGBRegressor()
clf = GridSearchCV(xgb_model,
                   {'max_depth': [2,4,6],
                    'n_estimators': [50,100,200]}, cv =5, verbose=1)

clf.fit(X,y)
print(clf.best_score_)
print(clf.best_params_)

