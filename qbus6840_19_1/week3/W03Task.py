#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:33:13 2019

@author: Boyan Zhang
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

dateparse = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y')

data = pd.read_csv('CBA_1991-2017.csv', parse_dates=['Date'], index_col='Date',date_parser=dateparse)
#print(data.head())
plt.plot(data)

#%%
Trend1 = data.rolling(2, center = True).mean().rolling(12,center = True).mean()
plt.plot(Trend1)

#%%
Trend2 = data.rolling(12,center = True).mean()
plt.plot(Trend2)
#%%
