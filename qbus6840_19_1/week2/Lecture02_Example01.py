#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 06:13:18 2017

@author: Professor Junbin Gao
"""

import numpy as np
import matplotlib.pyplot as plt
from drawnow import drawnow
import time

N = 200
t = np.arange(1,401,1) /200.0
y = 0.6*np.sin(t) #np.square(t)



s = np.random.randint(1,7,10)
s = s * np.array([1, -1, 1, 1, -1, -1, 1, 1, 1, -1])
s = np.tile(s, 40)

# We draw the pure trend
def draw_trend():
    plt.plot(t,y)
    plt.title('The Trend')

# The function to draw the trend plus reasonals
def draw_seasonal():
    plt.plot(y+s/50.0)
    plt.title('The Trend Plus Seasonals')
    
# The function to draw the simulated data
def draw_withNoise():
    plt.plot(y+s/50.0+0.05*np.random.normal(size=400))
    plt.title('A Synthetic Possibly True Time Series')

def draw_adjusted(): 
    plt.plot(y+0.01*np.random.normal(size=400))
    plt.title('Seasonally Adjusted Data!')

#plt.figure()
time.sleep(3)
drawnow(draw_trend)
time.sleep(5)  # Sleep three seconds
drawnow(draw_trend)
time.sleep(5)
drawnow(draw_seasonal)
time.sleep(20)
drawnow(draw_seasonal)
time.sleep(5)
drawnow(draw_withNoise)
time.sleep(5)
drawnow(draw_withNoise)
time.sleep(5)
drawnow(draw_adjusted)

