
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib.colors import ListedColormap 


data_test= pd.read_csv('Lecture5_Lin_Sep.csv') 
# try the following data as well, and see the impact of one addtional observation on the decision boundary of SVM
# data_test= pd.read_csv('Lecture5_Lin_Sep_1.csv') 

data_pos= [(data_test['y'] == 1)]

data_pos= data_test[(data_test.y== 1) ]
data_neg= data_test[(data_test.y== 0) ]

X = data_test.iloc[:, :-1]
X_1= data_pos.iloc[:, :-1]
X_0= data_neg.iloc[:, :-1]

y = data_test.iloc[:, -1]

linear_clf = svm.SVC(kernel = "linear", C=1000.0)
linear_clf.fit(X, y)
linear_clf.coef_
linear_clf.intercept_
linear_clf.support_

# Meshgrid resolution
h = .02

# Meshgrid colours
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

# Meshgrid boundaries
x_min, x_max = X[['x1']].min() - 1, X[['x1']].max() + 1
y_min, y_max = X[['x2']].min() - 1, X[['x2']].max() + 1

# Create the meshgrid coords
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Get predictions over the whole meshgrid
Z = linear_clf.predict(np.c_[xx.ravel(), yy.ravel()])  
Z = Z.reshape(xx.shape)

# Plot the mesh
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.title('Linear SVM')

# Plot the decision boundary
decis = linear_clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
decis = decis.reshape(xx.shape)
plt.contour(xx, yy, decis, levels=[0], linewidths=2, linetypes='--', colors='k')

plt.plot(X_1[['x1']], X_1[['x2']], 'o', color = 'b')
plt.plot(X_0[['x1']], X_0[['x2']], 'o', color = 'r')

