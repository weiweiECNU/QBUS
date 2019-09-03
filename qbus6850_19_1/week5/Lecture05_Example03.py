# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 14:24:34 2018

@author: Professor Junbin Gao
adotped from
http://scikit-learn.org/stable/auto_examples/exercises/plot_iris_exercise.html
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn import datasets, svm

iris = datasets.load_iris()
X = iris.data
y = iris.target

iris_train_df = pd.read_csv('iris.csv')

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X = iris_train_df.iloc[:,0:4].values

# Actually PCA does not need target information as it is an unsupervised learning
# We extract the target for visualisation purpose
t = iris_train_df.iloc[:,4]
le = preprocessing.LabelEncoder()
le.fit(t)
t = le.transform(t)
class_names = le.classes_


# We will only use two features and two classes
X = X[t != 0, :2]
t = t[t != 0]

n_sample = len(X)

# Randomly order the sample
np.random.seed(0)
order = np.random.permutation(n_sample)
X = X[order]
t = t[order].astype(np.float)

X_train = X[:int(.9 * n_sample)]
t_train = t[:int(.9 * n_sample)]
X_test = X[int(.9 * n_sample):]
t_test = t[int(.9 * n_sample):]

# fit the model

kernel = 'rbf'

clf = svm.SVC(kernel=kernel, gamma=10)
clf.fit(X_train, t_train)

fig_num = 1
plt.figure(fig_num)
plt.clf()
plt.scatter(X[:, 0], X[:, 1], c=t, zorder=10, cmap=plt.cm.Paired,
                edgecolor='k', s=20)

# Circle out the test data
plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none',
                zorder=10, edgecolor='k')

plt.axis('tight')
x_min = X[:, 0].min()
x_max = X[:, 0].max()
y_min = X[:, 1].min()
y_max = X[:, 1].max()

XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

Z = Z.reshape(XX.shape)    
plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
            linestyles=['--', '-', '--'], levels=[-.5, 0, .5])
plt.title(kernel)

   