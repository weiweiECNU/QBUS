#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 11:38:41 2018

@author: Professor Junbin Gao
adopted from
https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/

xgboost Installation Guide
https://xgboost.readthedocs.io/en/latest/build.html
"""
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")

# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


# fit model no training data
# Actually we follow the exactly same way as we do modelling with sklearn-kit
model = XGBClassifier()
model.fit(X_train, y_train)

print(model)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

	
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

F1_score = f1_score(y_test, predictions)
print("Accuracy: %.2f%%" % (F1_score * 100.0))
 