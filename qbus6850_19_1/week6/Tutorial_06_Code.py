# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 11:15:00 2018

@author: z3521729
"""

#%%
"""
Import external library
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib.colors import ListedColormap 

#%%
"""
Generate some non-linearly seperable data
"""
# 1 - Intialise the RNG
np.random.seed(1)

# 2 - Set the number of points to be created in each class
N = 100

# 3 - Generate the inner data
inner_mean = (0, 0)
inner_cov = [[10, 0], [0, 10]]
inner = np.random.multivariate_normal(inner_mean, inner_cov, N)

# 4 - Generate the outer data
dist_mean = 10
dist_var = 1
radius = dist_var * np.random.randn(N, 1) + dist_mean
angles = np.random.rand(N, 1) * (2*np.pi)
outer_x = radius * np.cos(angles)
outer_y = radius * np.sin(angles)
outer = np.concatenate( (outer_x, outer_y), axis = 1)

# 5 - Plot the data
fig0 = plt.figure()
plt.plot(inner[:,0], inner[:,1], 'o', color = 'b')
plt.plot(outer[:,0], outer[:,1], 'o', color = 'r')
fig0


#%%
"""
We then combine the two groups of points into a single matrix  X. 
Since the SVM classifier function expects a single data matrix. 
We also create a label or target vector  y
"""
X = np.concatenate( (inner, outer), axis = 0)
y = np.concatenate( (np.ones(N), np.zeros(N)) , axis = 0)


#%%
"""
Linear SVM
"""
# Fit the SVM classifier
linear_clf = svm.SVC(kernel = "linear")
linear_clf.fit(X, y)

# Plot the classifier results
fig1 = plt.figure()

# Meshgrid resolution
h = .02

# Meshgrid colours
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

# Meshgrid boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# Create the meshgrid coords
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Get predictions over the whole meshgrid
Z = linear_clf.predict(np.c_[xx.ravel(), yy.ravel()])  
Z = Z.reshape(xx.shape)

# Plot the mesh
fig1 = plt.figure()

plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.title('Linear SVM')

# Plot the decision boundary
decis = linear_clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
decis = decis.reshape(xx.shape)
plt.contour(xx, yy, decis, levels=[0], linewidths=2, linetypes='--', colors='k')

# Overlay the original points
plt.plot(inner[:,0], inner[:,1], 'o', color = 'b')
plt.plot(outer[:,0], outer[:,1], 'o', color = 'r')
fig1


#%%
"""
Kernel SVM
"""
kernel_clf = svm.SVC(kernel = "rbf")
kernel_clf.fit(X, y)

fig2 = plt.figure()

h = .02

# Get predictions over the whole meshgrid
Z = kernel_clf.predict(np.c_[xx.ravel(), yy.ravel()])  
Z = Z.reshape(xx.shape)

# Plot the mesh
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.title('Kernel SVM')

# Plot the decision boundary
decis_kernel = kernel_clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
decis_kernel = decis_kernel.reshape(xx.shape)
plt.contour(xx, yy, decis_kernel, levels=[0], linewidths=2, linetypes='--', colors='k')

# Overlay the original points
plt.plot(inner[:,0], inner[:,1], 'o', color = 'b')
plt.plot(outer[:,0], outer[:,1], 'o', color = 'r')

fig2


#%%
"""
Kernel SVM - Face Recognition of Famous People
"""

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from time import time
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

"""
Preparing data for training and testing
"""

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

n_samples, h, w = lfw_people.images.shape

# Use the data vectors directly (as relative pixel positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# The label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: {0}".format(n_samples))
print("n_features: {0}".format(n_features))
print("n_classes: {0}".format(n_classes))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

"""
Computing the new feature representation (eigenfaces)
"""
n_components = 150

print("Extracting the top {0} eigenfaces from {1} faces".format(n_components, X_train.shape[0]))
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)


"""
Optimising Parameters via Cross Validation
"""
print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("Done in {0:.3f}s".format((time() - t0)))
print("Best estimator found")


"""
Evaluating Classification Accuracy
"""
print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("Done in {0:.3f}s".format((time() - t0)))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


"""
Visualising Classification Results
"""
def plot_gallery(images, titles, h, w, n_row, n_col):
    """Helper function to plot a gallery of portraits"""
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return "Predicted: {0}\ntrue:      {1}".format(pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

n_col = 4
n_row = 3
fig3 = plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
plot_gallery(X_test, prediction_titles, h, w, n_row, n_col)
fig3

# Plot eigenface
eigenface_titles = ["Eigenface {0}".format(i) for i in range(eigenfaces.shape[0])]

fig4 = plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
plot_gallery(eigenfaces, eigenface_titles, h, w, n_row, n_col)
fig4



