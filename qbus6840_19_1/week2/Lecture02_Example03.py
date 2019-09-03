# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 08:31:51 2017

@author: Professor Junbin Gao
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from drawnow import drawnow
import time

data = pd.read_csv('beer.csv')

beer = data['x']

def draw_rolling_mean():
    plt.plot(beer)
    plt.plot(rolling_mean, '-r')
    title_str = 'Moving Average (window size = ' + str(i) + ')'
    plt.title(title_str)
    plt.xlim((1,55))

all_means = np.repeat(np.NaN, 56)     
for i in np.arange(3,27,1):
    r = beer.rolling(window=i, center=True) # We define the rolling window size
                                            # center = True means smoothing
                                            # center = False (default) means prediction 
    rolling_mean = r.mean()  
    all_means = np.vstack((all_means, rolling_mean))
    time.sleep(3)
    drawnow(draw_rolling_mean)
    

plt.figure() 
plt.plot(beer)
plt.plot(all_means[3,:],'-y')    # for k = 3
plt.plot(all_means[7,:],'-r')    # for k = 7