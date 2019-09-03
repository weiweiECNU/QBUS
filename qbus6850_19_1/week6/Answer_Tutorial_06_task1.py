#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 11:30:41 2018

@author: steve
"""

import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

bank_df = pd.read_csv("bank.csv")

# Find out how many positive cases in dataset
n_pos = sum(bank_df['y_yes'])
n_neg = len(bank_df) - n_pos

print("Proportion of positive responses {0:.2f}".format(n_pos/n_neg))

X = bank_df.iloc[:, 0:-1]
y = bank_df['y_yes']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Build an equally weighted SVM classifier
clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Reweight for positive class
clf_weighted = svm.SVC(class_weight='balanced')
clf_weighted.fit(X_train, y_train)
y_pred_weighted = clf_weighted.predict(X_test)

print(classification_report(y_test, y_pred_weighted))
print(confusion_matrix(y_test, y_pred_weighted))
