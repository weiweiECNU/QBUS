# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 13:01:04 2018

@author: Professor Junbin Gao
adopted from
http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html#sphx-glr-auto-examples-cluster-plot-cluster-iris-py
"""

import numpy as np
import matplotlib.pyplot as plt
# Though the following import is not directly being used, it is required
# for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans 
from sklearn import preprocessing
import pandas as pd

np.random.seed(5)

iris_train_df = pd.read_csv('iris.csv')

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X = iris_train_df.iloc[:,0:4].values

#We convert literal values into label values
t = iris_train_df.iloc[:,4]
le = preprocessing.LabelEncoder()
le.fit(t)
t = le.transform(t)
class_names = le.classes_

# k_means_iris_8 with good initialisation strategy provided by KMeans
est1 = KMeans(n_clusters=8)
est1.fit(X)
labels = est1.labels_

fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float), edgecolor='k')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.dist = 12
ax.set_title('8 clusters')

# k_means_iris_3 with good initialisation strategy provided by KMeans
est2 = KMeans(n_clusters=3)
est2.fit(X)
labels = est2.labels_

fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float), edgecolor='k')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.dist = 12
ax.set_title('3 clusters')

# k_means_iris_3 with bad initialisation, starting a random initialistion once
est3 = KMeans(n_clusters=3,  n_init = 1, init='random')
est3.fit(X)
labels = est3.labels_

fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float), edgecolor='k')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.dist = 12
ax.set_title('3 clusters, bad initialization')

# Plot the ground truth
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

for name, label in [('Setosa', 0),
                    ('Versicolour', 1),
                    ('Virginica', 2)]:
    ax.text3D(X[t == label, 3].mean(),
              X[t == label, 0].mean(),
              X[t == label, 2].mean() + 2, name,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
 
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=t, edgecolor='k')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.set_title('Ground Truth')
ax.dist = 12

 