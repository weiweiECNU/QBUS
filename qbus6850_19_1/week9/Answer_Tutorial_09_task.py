# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:03:36 2019

@author: Boyan Zhang
"""

#%%Task 1
import numpy as np
import pandas as pd
from random import shuffle
from sklearn import datasets

np.random.seed(5)

iris = datasets.load_iris()
X = iris.data
t = iris.target
N = len(t)
index = list(range(N))
shuffle(index)

X1 = X[index]
t1 = t[index]

#%%
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

learning_rate = [ 0.001, 0.01, 0.1]
param_test = {'n_estimators':range(10,30,10),
              'max_depth':range(3,5,1),
              'learning_rate':learning_rate}

model = GradientBoostingClassifier()
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid=param_test, scoring="neg_log_loss",  cv=kfold)
grid_result = grid_search.fit(X1, t)

#%%
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))

