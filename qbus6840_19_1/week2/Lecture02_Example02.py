#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 19:33:12 2017

@author: steve and junbin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from drawnow import drawnow
import time

 


beer_df = pd.read_csv('beer.txt', header=None)

#figure1 = plt.figure()
#plt.plot(beer_df)
#plt.title('The Beer Consumption Data')

beer_extra = np.concatenate((beer_df[0].values, [np.NaN, np.NaN, np.NaN, np.NaN]), 0)

# each row is a year
yearly_data = np.reshape(beer_extra, (5,12))

figure2 = plt.figure()
plt.plot(beer_extra)
plt.title('The Beer Consumption Data')
plt.xlim([0, 60])
plt.ylim([110, 200])

data = beer_extra.copy()

def draw_yearly():
    remove = range((year)*12, (year+1)*12, 1)
    data[remove] = np.NaN
    plt.plot(data)
    for d in range(1,year,1):
        plt.plot(yearly_data[d])
    xaxis = range((year)*12+1-step, (year+1)*12 - step + 1, 1)
    plt.plot(xaxis, yearly_data[year])
    plt.xlim([0, 60])
    plt.ylim([110, 200])
    plt.title('Year ' + str(year+1))

def draw_yearly_last():
    remove = range((year)*12, (year+1)*12, 1)
    data[remove] = np.NaN
    plt.plot(data)
    for d in range(1,year,1):
        plt.plot(yearly_data[d])
    xaxis = range((year)*12+1-step, (year+1)*12 - step + 1, 1)
    plt.plot(xaxis, yearly_data[year])
    plt.xlim([0, 60-step])
    plt.ylim([110, 200])
    if step == 49:
       plt.title('All Five Years')
    else:
       plt.title('Year ' + str(year+1))    

for j in range(1,4,1):
    year = j
    for i in range(12*year+1): 
        step = i+1
        drawnow(draw_yearly)
        time.sleep(0.2)
 
for i in range(4*12+1):
    year = 4
    step = i + 1
    drawnow(draw_yearly_last)
    time.sleep(0.2)
 

 

