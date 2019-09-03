# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 09:35:15 2017
Revised on Fri Mar 9  09:17 2018

@author: Professor Junbin Gao
"""
 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
 
# Prepare data
# Or read data from a file
x = np.linspace(0, 2*np.pi, 3000)
d = (2+np.random.normal(0, 1, 3000)) * np.c_[np.sin(x), np.cos(x)].T

# Prepare colors and lengends
lg = []
colors = (i + j for j in 'o<.' for i in 'bgrcmyk')
labels = 'one two three four five six seven eight nine ten'.split()


plt.figure()   # Open a window to draw

for i, l, c  in zip(range(10), labels, colors):
    start, stop = i * 300, (i + 1) * 300   # pick up 300 points from the series
                                           # label means to add lengends
    handle = plt.plot(d[0, start:stop], d[1, start:stop], c, label=l)
    lg.append(handle)
    plt.legend()

plt.show()

plt.figure()
plt.scatter(d[0, start:stop], d[1, start:stop]) 

data = pd.read_csv('beer.csv')

beer = data['x']

# We can directly draw pandas Series with matplotlib
# Note we get an axes to add grid lines
plt.figure()
ax = plt.axes()
ax.plot(beer)
plt.title('Monthly Beer Consumption 1991 - 1995')
plt.xlabel('Months')
plt.ylabel('Megalitres')
ax.yaxis.grid()
plt.xlim([0,55])

plt.figure()
ax = plt.axes()
ax.scatter(range(0,56),beer)
plt.title('Monthly Beer Consumption 1991 - 1995 (Actual Time Series)')
plt.xlabel('Months')
plt.ylabel('Megalitres')
ax.yaxis.grid()
plt.xlim([0,55])