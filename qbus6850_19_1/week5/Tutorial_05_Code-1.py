#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 16:23:01 2017

@author: steve
"""

"""

kNN Classification

k Nearest Neighbors is a very simple, yet highly effective and scalable
classifcation method. It is supervised in the sense that we build a decision
mesh based on observed training data. It is non-parametric, meaning
it makes no assumptions about the structure or form of the generative function.
It is also instance-based which means that it doesn't explicitiy learn
a model. Instead it simply uses the training data as "knowledge" about the
domain. Practically this means that the training data is only used when a
specific query is made ("which data points am I closest to?").

This highly flexible and "just in time" nature means that it scales well to
hugre volumes of data and is often the first port of call for classifying
massive data sets.

Since kNN is non-parametric it is generally considered to have high variance
and low bias. However we can tune this by increasing k - the number of neighbours
to average over. A higher value for k leads to a smoother decision mesh/boundary
and lower variance. For each application you must tune k on your training data
for optimal performance.

kNN can perform poorly if your features are not all at the same scale.
Therefore it is good to normalise them to be within the same range, otherwise
features with larger or smaller scales can dominate.

"""


import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap 
from sklearn import neighbors
from sklearn.datasets import make_blobs

X, y = make_blobs(100, centers= [[2, 2],[1, -2]], cluster_std=[1.3, 1.3])


"""
Initialise some plotting variables
"""
h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


"""
Do KNN Classification for k = 1
"""
n_neighbor_1 = 1

knn_clf_1 = neighbors.KNeighborsClassifier(n_neighbor_1, weights='uniform' )
knn_clf_1.fit(X, y)

Z_1 = knn_clf_1.predict(np.c_[xx.ravel(), yy.ravel()])  
# np.c_ is a way to combine 1D arrays into 2D array
# Put the result into a color plot
Z_1 = Z_1.reshape(xx.shape)

"""
Do KNN Classification for k = 50
"""
n_neighbor_2 = 50

knn_clf_50 = neighbors.KNeighborsClassifier(n_neighbor_2, weights='uniform' )
knn_clf_50.fit(X, y)

Z_50 = knn_clf_50.predict(np.c_[xx.ravel(), yy.ravel()])  

Z_50 = Z_50.reshape(xx.shape)

plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.gcf().set_size_inches(10, 5)
plt.pcolormesh(xx, yy, Z_1, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("2-Class classification (k = {0})".format(n_neighbor_1))

plt.subplot(1, 2, 2)
plt.gcf().set_size_inches(10, 5)
plt.pcolormesh(xx, yy, Z_50, cmap=cmap_light)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("2-Class classification (k = {0})".format(n_neighbor_2))

#%%

"""

We can use kNN for all sorts of data. Here we will use kNN to classify
images of hand written English characters.

In this dataset there are 1797 character image. Each image is 8x8 pixels i.e.
of dimension 64. We reshape each image to a 64x1 vector. Therefore we are
"searching" in 64 dimensional space.

Here we also introduce the train, validation and test split.

We will train our model on the train set, determine it's performance on the
validation set and then perform final evaluation on the test set.

"""

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

digits = datasets.load_digits()

# Show the first image
fig6 = plt.figure()
plt.gray()
for i in range(1, 11):
    plt.subplot(1, 11, i)
    plt.imshow(digits.images[i - 1])
fig6

trainData, testData, trainLabels, testLabels = train_test_split(np.array(digits.data), digits.target, test_size=0.25, random_state=42)

print("m Training Points: {}".format(len(trainLabels)))
print("m Test points: {}".format(len(testLabels)))

# Store cv score for each k
cv_scores = []
k_vals = []
 
for k in range(1, 30, 2):
    
    model = neighbors.KNeighborsClassifier(n_neighbors=k)
    
    scores = cross_val_score(model, trainData, trainLabels, cv=10, scoring='accuracy')
    score = scores.mean()
    
    print("k={0}, cv_score={1:.2f}".format(k, score * 100))
    
    cv_scores.append(score)
    k_vals.append(k)
 

# Find best performing k
idx = np.argmax(cv_scores)
print("k={0} achieved highest accuracy of {1:.2f}".format(k_vals[idx], cv_scores[idx] * 100))

model = neighbors.KNeighborsClassifier(n_neighbors = k_vals[idx])

model.fit(trainData, trainLabels)

predictions = model.predict(testData)
 
# Final classification report
print(classification_report(testLabels, predictions))

print(confusion_matrix(testLabels, predictions))


#%%
"""

K-means Clustering

K-means is a very simple clustering method. It performs suprisingly well in
practice and its simplicity means that it can be applied to massive datasets
with ease.

K-means is unsupervised and non-parametric. It does not explicity prescribe
a model. However its avergaing method performs best under certain conditions,
such as Gaussian distributed clusters. It is also affected greatly by choice
of initial cluster centroids.

We will illustrate the following weaknesses of k-means:

    - Sensitivity to number of clusters. You must know before hand the true
    number of clusters in your data, otherwise it will perform poorly
    
    - Assumption of Gaussian distributed data. If data is anisotrpoically
    distributed (stretched) then k-means often fails.
    
    - Clusters should have equal variance
    
    - Clusters should have equal size in space
    
"""

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

"""
Incorrect number of clusters
"""
y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)

plt.figure(figsize=(12, 6))

plt.subplot(1,2,1)
plt.scatter(X[:, 0], X[:, 1], c=[cm.spectral(float(i) /10) for i in y])
plt.title("Ground Truth")

plt.subplot(1,2,2)
plt.scatter(X[:, 0], X[:, 1], c=[cm.spectral(float(i) /10) for i in y_pred])
plt.title("K-means Clusters - Incorrect Number of Clusters")

"""
Anisotropicly distributed data
"""

transformation = [[ 0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X, transformation)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)

plt.figure(figsize=(12, 6))

plt.subplot(1,2,1)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=[cm.spectral(float(i) /10) for i in y])
plt.title("Ground Truth - Anisotropicly Distributed Clusters")

plt.subplot(1,2,2)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=[cm.spectral(float(i) /10) for i in y_pred])
plt.title("K-means Clusters - Anisotropicly Distributed Clusters")

"""
Clusters with unequal variance
"""
X_varied, y_varied = make_blobs(n_samples=n_samples,
                                cluster_std=[1.0, 2.5, 0.5],
                                random_state=random_state)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)

plt.figure(figsize=(12, 6))

plt.subplot(1,2,1)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=[cm.spectral(float(i) /10) for i in y_varied])
plt.title("Ground Truth - Unequal Variance")

plt.subplot(1,2,2)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=[cm.spectral(float(i) /10) for i in y_pred])
plt.title("K-means Clusters - Unequal Variance")

"""
Clusters with unequal size
"""
X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
y_filtered = np.vstack((np.zeros((500,1)), np.ones((100,1)), 2*np.ones((10,1))))

y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_filtered)

plt.figure(figsize=(12, 6))

plt.subplot(1,2,1)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=[cm.spectral(float(i) /10) for i in y_filtered])
plt.title("Ground Truth - Unevenly Sized Clusters")

plt.subplot(1,2,2)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=[cm.spectral(float(i) /10) for i in y_pred])
plt.title("K-means Clusters - Unevenly Sized Clusters")



