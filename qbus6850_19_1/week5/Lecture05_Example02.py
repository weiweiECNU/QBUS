
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib.colors import ListedColormap 
###################################################################
# linearly non-seperable data
data_non_sep= pd.read_csv('Lecture5_Lin_NonSep.csv')

data_pos_non_sep= data_non_sep[(data_non_sep.y== 1) ]
data_neg_non_sep= data_non_sep[(data_non_sep.y== 0) ]

X_non_sep = data_non_sep.iloc[:, :-1]
X_pos_non_sep= data_pos_non_sep.iloc[:, :-1]
X_neg_non_sep= data_neg_non_sep.iloc[:, :-1]

y_non_sep = data_non_sep.iloc[:, -1]

linear_non_sep = svm.SVC(kernel = "linear", C=1000.0)
linear_non_sep.fit(X_non_sep, y_non_sep)
linear_non_sep.coef_
linear_non_sep.intercept_
linear_non_sep.support_

# Meshgrid resolution
h = .02

# Meshgrid colours
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

# Meshgrid boundaries
x_min, x_max = X_non_sep[['x1']].min() - 1, X_non_sep[['x1']].max() + 1
y_min, y_max = X_non_sep[['x2']].min() - 1, X_non_sep[['x2']].max() + 1

# Create the meshgrid coords
xx_non_sep, yy_non_sep = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Get predictions over the whole meshgrid
Z = linear_non_sep.predict(np.c_[xx_non_sep.ravel(), yy_non_sep.ravel()])  
Z = Z.reshape(xx_non_sep.shape)

# Plot the mesh
plt.pcolormesh(xx_non_sep, yy_non_sep, Z, cmap=cmap_light)
plt.title('Linear SVM')

# Plot the decision boundary
decis = linear_non_sep.decision_function(np.c_[xx_non_sep.ravel(), yy_non_sep.ravel()])
decis = decis.reshape(xx_non_sep.shape)
plt.contour(xx_non_sep, yy_non_sep, decis, levels=[0], linewidths=2, linetypes='--', colors='k')

plt.plot(X_pos_non_sep[['x1']], X_pos_non_sep[['x2']], 'o', color = 'b')
plt.plot(X_neg_non_sep[['x1']], X_neg_non_sep[['x2']], 'o', color = 'r')

################################
# plot the decision boundary of slides 43&44
# Meshgrid resolution
h = .02

x1= np.linspace(-5,5,100)
x2= np.linspace(-5,5,100)

x_min, x_max = x1.min() - 1, x1.max() + 1
y_min, y_max = x2.min() - 1, x2.max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

f_xy= -5+xx.ravel()+xx.ravel()**2+yy.ravel()+yy.ravel()**2
# f_xy= -5+xx.ravel()+xx.ravel()**2+xx.ravel()**3+yy.ravel()+yy.ravel()**2+yy.ravel()**3
f_xy = f_xy.reshape(xx.shape)

plt.contour(xx, yy, f_xy, levels=[0], linewidths=2, linetypes='--', colors='k')
