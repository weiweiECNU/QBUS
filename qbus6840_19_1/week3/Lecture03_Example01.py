#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:31:10 2017
revised on Fri Mar 16 10:28:00 2018

@author: steve and junbin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from drawnow import drawnow
import time


beer_df = pd.read_csv('beer.txt', header=None)

beer_extra = np.concatenate( (beer_df[0].values, beer_df[0].values[8:12]), 0)

beer = np.concatenate( (beer_extra, beer_extra), 0).copy()

#data = beer.copy()
                

#figure1 = plt.figure()
#plt.plot(beer)
#plt.title('The Beer Consumption Data')

# rows = years, # cols = months
yearly_data = np.reshape(beer, (10,12)).copy()

Months = ['January','Febraury','March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

Steps = 12
line_specs = ['og-', 'or-', 'oc-', 'om-', 'oy-', 'ok-', 'og-', 'or-', 'oc-', 'om-', 'oy-', 'ok-']
 
def draw_beer():
    plt.plot(beer)
    plt.title('The Beer Consumption in 20 Years')
    
def draw_beer_month():
    global moves
    plt.plot(beer)
    plt.plot(moves, line_specs[month], markevery = marker_location, linewidth=2)
    plt.title(Months[month])

drawnow(draw_beer)
time.sleep(3)

for month in range(12):
    marker_location = []
    circ = np.repeat(np.NaN, 120) 
    if month == 0:
       moves = []
    else:
       moves = np.repeat(np.NaN, month)  
    for year in range(9): 
        xaxis = range(month, 120, 12)
        month_beer = beer[xaxis]
        print(year)
        gradient = month_beer[year+1] - month_beer[year]
        for i in range(Steps):
            step = (i)/Steps
            moves = np.append(moves, month_beer[year] + step * gradient)
            drawnow(draw_beer_month)
            time.sleep(0.1)
            if (i==Steps-1) and (year == 8):
               moves = np.append(moves,  month_beer[year+1])
               marker_location = xaxis
               drawnow(draw_beer_month)
               time.sleep(1)


ims = []
for i in range(12):
    for j in range(10):
        ims.append(plt.plot(np.linspace(0,j*12,j+1), yearly_data[0:j+1,i], line_specs[i]))

#im_ani = animation.ArtistAnimation(figure1, ims, interval=400, blit=False, repeat=False)

